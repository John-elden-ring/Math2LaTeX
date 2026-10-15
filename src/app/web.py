"""Interactive Web Application for Math2LaTeX.

Allows users to:
1. Drag-and-drop or upload photos of handwritten math notes.
2. View detected equation line crops side-by-side with transcribed LaTeX.
3. See live rendered math previews rendered via KaTeX.
4. Edit any LaTeX line with real-time math re-rendering.
5. Download the complete, compilable .tex file.
"""

from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List
import uuid

import base64
import cv2
from flask import Flask, Response, jsonify, render_template_string, request, send_file, send_from_directory
import numpy as np

from src.pipeline.page_to_tex import PageToTexPipeline, format_latex_document
from src.model.evaluate import check_latex_compilable


app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent.parent.parent / "static"))
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Global pipeline instance (lazy loaded)
PIPELINE_INSTANCE = None
UPLOAD_CACHE_DIR = Path("static/uploads")
CROPS_CACHE_DIR = Path("static/crops")
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CROPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def image_to_base64_url(img: np.ndarray) -> str:
    """Encode an image array directly into a data:image/png;base64 URL."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_pipeline():
    global PIPELINE_INSTANCE
    if PIPELINE_INSTANCE is None:
        # Check available checkpoints in priority order
        candidates = [
            "checkpoints/best_model.pt",
            "checkpoints/overfit_100_model.pt",
            "checkpoints/latest_model.pt",
        ]
        ckpt = "checkpoints/latest_model.pt"
        for c in candidates:
            if Path(c).exists():
                ckpt = c
                break
        print(f"Initializing Math2LaTeX pipeline with checkpoint: {ckpt}")
        PIPELINE_INSTANCE = PageToTexPipeline(checkpoint_path=ckpt)
    return PIPELINE_INSTANCE


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Math2LaTeX — Handwritten Math to LaTeX Converter</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>

  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #e2e8f0;
      --border-focus: #2563eb;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #0f172a;
      --primary-hover: #1e293b;
      --secondary-bg: #f1f5f9;
      --secondary-hover: #e2e8f0;
      --accent-blue: #2563eb;
      --success: #15803d;
      --warning: #b45309;
      --danger: #b91c1c;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Navigation Bar */
    .navbar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
    }
    .navbar-inner {
      max-width: 1080px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-title {
      font-size: 1.125rem;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.01em;
    }
    .brand-sub {
      font-size: 0.8125rem;
      color: var(--text-muted);
      border-left: 1px solid var(--border);
      padding-left: 10px;
    }
    .model-status {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.8125rem;
      color: var(--text-muted);
      background: var(--secondary-bg);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #16a34a;
    }

    /* Main Container */
    .container {
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 80px;
    }

    /* Page Intro */
    .intro {
      margin-bottom: 24px;
    }
    .intro h1 {
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 6px;
      letter-spacing: -0.01em;
    }
    .intro p {
      color: var(--text-muted);
      font-size: 0.9375rem;
    }

    /* Upload Zone */
    .upload-zone {
      background: var(--surface);
      border: 1.5px dashed #cbd5e1;
      border-radius: 8px;
      padding: 40px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.15s, background-color 0.15s;
      margin-bottom: 32px;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent-blue);
      background-color: #f8fafc;
    }
    .upload-icon {
      margin-bottom: 12px;
      color: #64748b;
    }
    .upload-title {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 4px;
    }
    .upload-hint {
      font-size: 0.8125rem;
      color: var(--text-muted);
      margin-bottom: 16px;
    }
    .upload-actions {
      display: flex;
      justify-content: center;
      gap: 12px;
      align-items: center;
    }

    /* Standard Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-sans);
      font-size: 0.875rem;
      font-weight: 500;
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      border: 1px solid transparent;
      text-decoration: none;
      transition: background-color 0.15s, border-color 0.15s;
    }
    .btn-primary {
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
    }
    .btn-primary:hover {
      background: var(--primary-hover);
    }
    .btn-secondary {
      background: var(--surface);
      color: var(--text);
      border-color: var(--border);
    }
    .btn-secondary:hover {
      background: var(--secondary-bg);
      border-color: #cbd5e1;
    }
    .btn-sm {
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 4px;
    }

    /* Loading Box */
    .loading-box {
      display: none;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 36px 24px;
      text-align: center;
      margin-bottom: 32px;
    }
    .spinner {
      width: 28px;
      height: 28px;
      border: 2px solid #e2e8f0;
      border-top-color: var(--text);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin: 0 auto 14px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-title {
      font-size: 0.9375rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 4px;
    }
    .loading-subtitle {
      font-size: 0.8125rem;
      color: var(--text-muted);
    }

    /* Results Section */
    .results-container {
      display: none;
    }
    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .results-title {
      font-size: 1.125rem;
      font-weight: 600;
      color: var(--text);
    }
    .results-count {
      font-size: 0.8125rem;
      color: var(--text-muted);
      margin-left: 8px;
      font-weight: 400;
    }
    .actions-bar {
      display: flex;
      gap: 8px;
    }

    /* Equation Card */
    .line-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 16px;
    }
    .line-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid #f1f5f9;
    }
    .line-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--text);
    }
    .line-meta {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .pill {
      font-size: 0.75rem;
      font-weight: 500;
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid transparent;
    }
    .pill-muted {
      background: var(--secondary-bg);
      color: var(--text-muted);
      border-color: var(--border);
    }
    .pill-valid {
      background: #f0fdf4;
      color: var(--success);
      border-color: #bbf7d0;
    }
    .pill-warn {
      background: #fffbeb;
      color: var(--warning);
      border-color: #fde68a;
    }

    /* Side-by-side view */
    .line-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 14px;
    }
    @media (max-width: 768px) {
      .line-grid { grid-template-columns: 1fr; }
    }
    .pane-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .crop-preview-box {
      background: #fafafa;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 84px;
    }
    .crop-preview-box img {
      max-width: 100%;
      max-height: 100px;
      object-fit: contain;
    }
    .math-render-box {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 84px;
      overflow-x: auto;
      font-size: 1.25rem;
      color: #000000;
    }

    /* LaTeX Editor Area */
    .latex-input-container {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .input-bar {
      display: flex;
      gap: 8px;
    }
    .latex-input {
      flex: 1;
      font-family: var(--font-mono);
      font-size: 0.875rem;
      color: var(--text);
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 12px;
      outline: none;
    /* Sample Gallery Card */
    .sample-picker-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 32px;
    }
    .sample-picker-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 12px;
    }
    .sample-picker-title {
      font-size: 0.9375rem;
      font-weight: 600;
      color: var(--text);
    }
    .sample-picker-sub {
      font-size: 0.8125rem;
      color: var(--text-muted);
    }
    .sample-picker-controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .sample-dropdown {
      flex: 1;
      min-width: 260px;
      font-family: var(--font-sans);
      font-size: 0.875rem;
      color: var(--text);
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 12px;
      outline: none;
      cursor: pointer;
    }
    .sample-dropdown:focus {
      border-color: var(--border-focus);
    }
    .sample-ref-banner {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 16px;
      margin-bottom: 16px;
    }

    /* Notification Toast */
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--primary);
      color: #ffffff;
      padding: 10px 16px;
      border-radius: 6px;
      font-size: 0.875rem;
      font-weight: 500;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      display: none;
      z-index: 1000;
    }
  </style>
</head>
<body>
  <header class="navbar">
    <div class="navbar-inner">
      <div class="brand">
        <span class="brand-title">Math2LaTeX</span>
        <span class="brand-sub">Handwritten Math Transcriber</span>
      </div>
      <div class="model-status">
        <span class="status-dot"></span>
        <span>GPU Model Active</span>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="intro">
      <h1>Handwritten Math to LaTeX</h1>
      <p>Upload a photo of handwritten equations. The pipeline segments individual lines, runs OCR, and generates compilable LaTeX.</p>
    </section>

    <!-- Upload Area -->
    <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
      <div class="upload-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
      </div>
      <div class="upload-title">Choose an image or drop it here</div>
      <div class="upload-hint">PNG, JPG, or JPEG up to 32MB</div>
      <div class="upload-actions">
        <button type="button" class="btn btn-secondary" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">Browse files</button>
        <button type="button" class="btn btn-secondary" onclick="event.stopPropagation(); loadRandomSample()">🎲 Random sample (1 of 40)</button>
      </div>
    </div>

    <!-- 40 Sample Equation Gallery -->
    <div class="sample-picker-card">
      <div class="sample-picker-header">
        <div>
          <div class="sample-picker-title">Sample Equation Gallery (40 Available)</div>
          <div class="sample-picker-sub">Pick from 40 handwritten benchmark equations across 8 mathematical categories, or load randomly</div>
        </div>
        <button type="button" class="btn btn-primary" onclick="loadRandomSample()">🎲 Random Sample</button>
      </div>
      <div class="sample-picker-controls">
        <select id="sampleSelect" class="sample-dropdown" onchange="onSampleSelected(this.value)">
          <option value="">Loading 40 samples...</option>
        </select>
        <button type="button" class="btn btn-secondary" onclick="loadSelectedSample()">Run Selected</button>
      </div>
    </div>

    <!-- Processing State -->
    <div class="loading-box" id="loadingState">
      <div class="spinner"></div>
      <div class="loading-title">Processing handwritten notes...</div>
      <div class="loading-subtitle">Deskewing, segmenting lines, and generating LaTeX</div>
    </div>

    <!-- Results Section -->
    <section class="results-container" id="resultsSection">
      <div class="results-header">
        <div>
          <span class="results-title">Recognized Equations</span>
          <span class="results-count" id="summaryText">0 equations</span>
        </div>
        <div class="actions-bar">
          <button class="btn btn-secondary" onclick="copyFullDocument()">Copy LaTeX</button>
          <button class="btn btn-primary" onclick="downloadTexFile()">Download .tex</button>
        </div>
      </div>

      <div id="linesList"></div>
    </section>
  </main>

  <div class="toast" id="toast">Copied to clipboard</div>

  <script>
    let currentLineData = [];
    let allSamples = [];

    // Initialize 40 samples gallery on load
    window.addEventListener('DOMContentLoaded', () => {
      initSampleGallery();
    });

    async function initSampleGallery() {
      try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        allSamples = data.samples || [];
        populateSampleDropdown(allSamples);
      } catch (e) {
        console.error('Failed loading sample gallery:', e);
      }
    }

    function populateSampleDropdown(samples) {
      const sel = document.getElementById('sampleSelect');
      if (!sel) return;
      sel.innerHTML = '<option value="">Select an equation from 40 samples...</option>';

      const groups = {};
      samples.forEach(s => {
        if (!groups[s.category]) groups[s.category] = [];
        groups[s.category].push(s);
      });

      Object.entries(groups).forEach(([cat, list]) => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = `${cat} (${list.length})`;
        list.forEach(item => {
          const opt = document.createElement('option');
          opt.value = item.id;
          opt.textContent = `${item.name}`;
          optgroup.appendChild(opt);
        });
        sel.appendChild(optgroup);
      });
    }

    function onSampleSelected(val) {
      if (val) {
        fetchAndRenderSample(val);
      }
    }

    function loadSelectedSample() {
      const sel = document.getElementById('sampleSelect');
      const val = sel.value;
      if (!val) {
        loadRandomSample();
        return;
      }
      fetchAndRenderSample(val);
    }

    function loadRandomSample() {
      if (allSamples.length > 0) {
        const rand = allSamples[Math.floor(Math.random() * allSamples.length)];
        const sel = document.getElementById('sampleSelect');
        if (sel) sel.value = rand.id;
        fetchAndRenderSample(rand.id);
      } else {
        fetchAndRenderSample('');
      }
    }

    async function fetchAndRenderSample(sampleId) {
      showLoading(true);
      try {
        const url = sampleId ? `/api/sample?id=${encodeURIComponent(sampleId)}` : '/api/sample?random=true';
        const res = await fetch(url);
        const data = await res.json();
        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }
        if (data.sample && data.sample.id) {
          const sel = document.getElementById('sampleSelect');
          if (sel) sel.value = data.sample.id;
        }
        renderResults(data);
      } catch (err) {
        alert('Failed loading sample: ' + err);
      } finally {
        showLoading(false);
      }
    }

    // Drag & Drop
    const dropZone = document.getElementById('dropZone');
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
    });
    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });

    function handleFileSelect(event) {
      if (event.target.files.length) {
        processUploadedFile(event.target.files[0]);
      }
    }

    async function processUploadedFile(file) {
      showLoading(true);
      const formData = new FormData();
      formData.append('photo', file);

      try {
        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }
        renderResults(data);
      } catch (err) {
        alert('Failed to process image: ' + err);
      } finally {
        showLoading(false);
      }
    }

    function showLoading(show) {
      document.getElementById('loadingState').style.display = show ? 'block' : 'none';
      if (show) {
        document.getElementById('resultsSection').style.display = 'none';
      }
    }

    function renderResults(data) {
      currentLineData = data.lines || [];
      document.getElementById('summaryText').textContent = `${currentLineData.length} equation${currentLineData.length === 1 ? '' : 's'}`;
      const listEl = document.getElementById('linesList');
      listEl.innerHTML = '';

      // If this was loaded from a sample, display reference banner
      if (data.sample) {
        const banner = document.createElement('div');
        banner.className = 'sample-ref-banner';
        banner.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
              <span class="pill pill-muted" style="margin-right: 6px;">${escapeHtml(data.sample.category)}</span>
              <strong style="font-size: 0.9375rem;">${escapeHtml(data.sample.name)}</strong>
            </div>
            <button type="button" class="btn btn-secondary btn-sm" onclick="loadRandomSample()">🎲 Next Random Sample &rarr;</button>
          </div>
          ${data.sample.ground_truth ? `
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;">
            <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">Dataset Ground Truth:</span>
            <code style="font-family: var(--font-mono); font-size: 0.8125rem; background: #ffffff; padding: 2px 6px; border-radius: 4px; border: 1px solid #cbd5e1; color: var(--text);">${escapeHtml(data.sample.ground_truth)}</code>
          </div>` : ''}
        `;
        listEl.appendChild(banner);
      }

      currentLineData.forEach((line, idx) => {
        const card = document.createElement('div');
        card.className = 'line-card';
        card.innerHTML = `
          <div class="line-header">
            <span class="line-title">Equation #${idx + 1}</span>
            <div class="line-meta">
              <span class="pill pill-muted">${Math.round(line.confidence * 100)}% confidence</span>
              <span class="pill ${line.compiles ? 'pill-valid' : 'pill-warn'}" id="tag-syntax-${idx}">
                ${line.compiles ? 'Valid syntax' : 'Check syntax'}
              </span>
            </div>
          </div>
          <div class="line-grid">
            <div>
              <div class="pane-label">Detected Line Crop</div>
              <div class="crop-preview-box">
                <img src="${line.crop_url}" alt="Line ${idx + 1}">
              </div>
            </div>
            <div>
              <div class="pane-label">KaTeX Preview</div>
              <div class="math-render-box" id="katex-${idx}"></div>
            </div>
          </div>
          <div class="latex-input-container">
            <div class="pane-label">LaTeX Code</div>
            <div class="input-bar">
              <input type="text" class="latex-input" id="input-${idx}" value="${escapeHtml(line.latex)}" oninput="updateMath(${idx}, this.value)">
              <button type="button" class="btn btn-secondary btn-sm" onclick="copySingleLine(${idx})">Copy</button>
            </div>
          </div>
        `;
        listEl.appendChild(card);
        renderKaTeX(idx, line.latex);
      });

      document.getElementById('resultsSection').style.display = 'block';
    }

    function autoBalanceLatex(s) {
      if (!s) return "";
      let str = s.trim();
      let cleanChars = [];
      let depth = 0;
      for (let ch of str) {
        if (ch === '{') {
          depth++;
          cleanChars.push(ch);
        } else if (ch === '}') {
          if (depth > 0) {
            depth--;
            cleanChars.push(ch);
          }
        } else {
          cleanChars.push(ch);
        }
      }
      if (depth > 0) cleanChars.push('}'.repeat(depth));
      str = cleanChars.join('');

      const numLeft = (str.match(/\\left(?![a-zA-Z])/g) || []).length;
      const numRight = (str.match(/\\right(?![a-zA-Z])/g) || []).length;
      if (numLeft > numRight) {
        let diff = numLeft - numRight;
        for (let i = 0; i < diff; i++) {
          const lastBareParen = str.lastIndexOf(')');
          if (lastBareParen > 0 && !str.substring(Math.max(0, lastBareParen - 6), lastBareParen).endsWith('\\right')) {
            str = str.substring(0, lastBareParen) + '\\right)' + str.substring(lastBareParen + 1);
          } else {
            str += '\\right.';
          }
        }
      }
      return str;
    }

    function renderKaTeX(idx, latex) {
      const el = document.getElementById(`katex-${idx}`);
      const tag = document.getElementById(`tag-syntax-${idx}`);
      if (!el) return;
      const cleanLatex = autoBalanceLatex(latex);
      try {
        katex.render(cleanLatex, el, {
          throwOnError: true,
          displayMode: true,
        });
        if (tag) {
          tag.className = 'pill pill-valid';
          tag.textContent = 'Valid syntax';
        }
      } catch (e) {
        try {
          katex.render(cleanLatex, el, {
            throwOnError: false,
            displayMode: true,
          });
          if (tag) {
            tag.className = 'pill pill-warn';
            tag.textContent = 'Auto-closed';
          }
        } catch (e2) {
          el.innerHTML = `
            <div style="font-size: 0.8125rem; color: #b91c1c; text-align: left; width: 100%;">
              <strong>Syntax Error:</strong> ${escapeHtml(e.message)}
            </div>
          `;
          if (tag) {
            tag.className = 'pill pill-warn';
            tag.textContent = 'Invalid syntax';
          }
        }
      }
    }

    function updateMath(idx, newLatex) {
      if (currentLineData[idx]) {
        currentLineData[idx].latex = newLatex;
      }
      renderKaTeX(idx, newLatex);
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function copySingleLine(idx) {
      const inputEl = document.getElementById(`input-${idx}`);
      if (inputEl) {
        navigator.clipboard.writeText(inputEl.value).then(() => {
          showToast('Equation LaTeX copied');
        });
      }
    }

    async function downloadTexFile() {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lines: currentLineData }),
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'notes.tex';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }

    async function copyFullDocument() {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lines: currentLineData, format: 'text' }),
      });
      const text = await res.text();
      navigator.clipboard.writeText(text).then(() => {
        showToast('Document LaTeX copied');
      });
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.display = 'block';
      setTimeout(() => { t.style.display = 'none'; }, 2000);
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


SAMPLE_GALLERY_FILE = Path("data/sample_gallery.json")


def get_sample_gallery() -> List[Dict[str, Any]]:
    """Load or generate the 40-item sample equation gallery."""
    if not SAMPLE_GALLERY_FILE.exists():
        try:
            from scripts.create_sample_gallery import main as create_gallery
            create_gallery()
        except Exception as e:
            print(f"Error auto-creating gallery: {e}")
            return []
    try:
        with open(SAMPLE_GALLERY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading gallery: {e}")
        return []


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return catalog of 40 available benchmark samples."""
    gallery = get_sample_gallery()
    return jsonify({
        "total": len(gallery),
        "samples": [
            {
                "id": s["id"],
                "name": s["name"],
                "category": s["category"],
                "ground_truth": s.get("ground_truth", ""),
                "latex_display": s.get("latex_display", ""),
            }
            for s in gallery
        ],
    })


@app.route("/api/upload", methods=["POST"])
def upload_page():
    if "photo" not in request.files:
        return jsonify({"error": "No photo file provided"}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image to temp cache
    session_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".jpg"
    upload_path = UPLOAD_CACHE_DIR / f"{session_id}{ext}"
    file.save(str(upload_path))

    pipeline = get_pipeline()
    tex_content, line_details, line_crops = pipeline.process_page(upload_path)

    # Save crop previews to static folder
    response_lines = []
    for detail, crop in zip(line_details, line_crops):
        crop_filename = f"{session_id}_crop_{crop.index}.png"
        crop_path = CROPS_CACHE_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop.image)

        response_lines.append({
            "index": detail["index"],
            "crop_url": image_to_base64_url(crop.image),
            "latex": detail["latex"],
            "confidence": detail["confidence"],
            "compiles": detail["compiles"],
        })

    return jsonify({
        "session_id": session_id,
        "lines": response_lines,
        "tex_content": tex_content,
    })


@app.route("/api/sample", methods=["GET"])
def get_sample():
    """Run transcription on a selected or randomized sample from the 40-item gallery."""
    gallery = get_sample_gallery()
    if not gallery:
        return jsonify({"error": "No sample gallery available"}), 404

    sample_id = request.args.get("id")
    is_random = request.args.get("random", "").lower() in ("true", "1") or not sample_id

    selected_sample = None
    if sample_id and not is_random:
        for s in gallery:
            if s["id"] == sample_id:
                selected_sample = s
                break

    if selected_sample is None:
        import random
        selected_sample = random.choice(gallery)

    img_path = Path(selected_sample["image_path"])
    if not img_path.exists():
        return jsonify({"error": f"Sample image file not found: {img_path}"}), 404

    page_img = cv2.imread(str(img_path))
    if page_img is None:
        return jsonify({"error": "Failed to read sample image"}), 500

    session_id = f"sample_{selected_sample['id']}_{int(time.time())}"
    pipeline = get_pipeline()
    tex_content, line_details, line_crops = pipeline.process_page(page_img)

    response_lines = []
    for detail, crop in zip(line_details, line_crops):
        crop_filename = f"{session_id}_crop_{crop.index}.png"
        crop_path = CROPS_CACHE_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop.image)

        response_lines.append({
            "index": detail["index"],
            "crop_url": image_to_base64_url(crop.image),
            "latex": detail["latex"],
            "confidence": detail["confidence"],
            "compiles": detail["compiles"],
        })

    return jsonify({
        "session_id": session_id,
        "sample": {
            "id": selected_sample["id"],
            "name": selected_sample["name"],
            "category": selected_sample["category"],
            "ground_truth": selected_sample.get("ground_truth", ""),
            "latex_display": selected_sample.get("latex_display", ""),
        },
        "lines": response_lines,
        "tex_content": tex_content,
    })


@app.route("/api/export", methods=["POST"])
def export_tex():
    data = request.get_json() or {}
    lines = data.get("lines", [])
    format_type = data.get("format", "file")

    tex_doc = format_latex_document(lines)

    if format_type == "text":
        return Response(tex_doc, mimetype="text/plain")

    buffer = BytesIO()
    buffer.write(tex_doc.encode("utf-8"))
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="notes.tex",
        mimetype="application/x-tex",
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch Math2LaTeX Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=5000, help="Port")
    args = parser.parse_args()

    print(f"Starting Math2LaTeX Web App on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
