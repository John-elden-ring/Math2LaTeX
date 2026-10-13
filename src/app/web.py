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

import cv2
from flask import Flask, Response, jsonify, render_template_string, request, send_file
import numpy as np

from src.pipeline.page_to_tex import PageToTexPipeline, format_latex_document
from src.model.evaluate import check_latex_compilable


app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Global pipeline instance (lazy loaded)
PIPELINE_INSTANCE = None
UPLOAD_CACHE_DIR = Path("static/uploads")
CROPS_CACHE_DIR = Path("static/crops")
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CROPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
  <title>Math2LaTeX — Handwritten Math Notes to LaTeX</title>
  <!-- Google Fonts & KaTeX for live rendering -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>

  <style>
    :root {
      --bg-dark: #0a0d14;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-purple: #8b5cf6;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --glow: rgba(59, 130, 246, 0.25);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.10) 0%, transparent 45%);
    }

    /* Container */
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 24px;
    }

    /* Header */
    header {
      text-align: center;
      margin-bottom: 40px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      background: rgba(59, 130, 246, 0.12);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
      margin-bottom: 16px;
    }
    h1 {
      font-size: 2.75rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 12px;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1.1rem;
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.6;
    }

    /* Upload Zone */
    .upload-card {
      background: var(--card-bg);
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: 20px;
      padding: 48px 32px;
      text-align: center;
      cursor: pointer;
      backdrop-filter: blur(12px);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      margin-bottom: 40px;
    }
    .upload-card:hover, .upload-card.dragover {
      border-color: var(--accent-blue);
      box-shadow: 0 0 35px var(--glow);
      transform: translateY(-2px);
    }
    .upload-icon {
      width: 64px;
      height: 64px;
      margin: 0 auto 16px;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
    }
    .upload-card h3 {
      font-size: 1.3rem;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .upload-card p {
      color: var(--text-muted);
      font-size: 0.95rem;
    }
    .btn-sample {
      display: inline-block;
      margin-top: 18px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: var(--text-main);
      padding: 8px 18px;
      border-radius: 10px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn-sample:hover {
      background: rgba(255, 255, 255, 0.12);
    }

    /* Loading Spinner */
    .loading-state {
      display: none;
      text-align: center;
      padding: 30px;
    }
    .spinner {
      width: 44px;
      height: 44px;
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top-color: var(--accent-cyan);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin: 0 auto 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Results Section */
    .results-container {
      display: none;
    }
    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .results-header h2 {
      font-size: 1.6rem;
      font-weight: 600;
    }
    .actions-bar {
      display: flex;
      gap: 12px;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 20px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      border: none;
      transition: all 0.2s;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      color: white;
      box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .btn-primary:hover {
      opacity: 0.95;
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-main);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.14);
    }

    /* Equation Line Cards */
    .line-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 20px;
      backdrop-filter: blur(10px);
      transition: border-color 0.2s;
    }
    .line-card:hover {
      border-color: rgba(59, 130, 246, 0.3);
    }
    .line-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .line-title {
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text-muted);
    }
    .line-meta {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .tag {
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .tag-conf {
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tag-syntax {
      background: rgba(59, 130, 246, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .tag-warn {
      background: rgba(245, 158, 11, 0.15);
      color: var(--warning);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Line Grid */
    .line-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 800px) {
      .line-grid { grid-template-columns: 1fr; }
    }
    .crop-preview-box {
      background: #ffffff;
      border-radius: 12px;
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 80px;
      overflow: hidden;
    }
    .crop-preview-box img {
      max-width: 100%;
      max-height: 90px;
      object-fit: contain;
    }
    .math-render-box {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 80px;
      overflow-x: auto;
      font-size: 1.25rem;
    }
    .latex-input-box {
      margin-top: 14px;
    }
    .latex-input {
      width: 100%;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      padding: 12px 14px;
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .latex-input:focus {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(6, 182, 212, 0.25);
    }

    /* Toast Notification */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #1e293b;
      color: white;
      padding: 12px 20px;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      display: none;
      z-index: 100;
      font-weight: 500;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">Custom Trained ResNet + Transformer</div>
      <h1>Math2LaTeX Transcriber</h1>
      <p class="subtitle">Upload a photo of handwritten equations to automatically extract, transcribe, and compile them into clean LaTeX code.</p>
    </header>

    <!-- Upload Card -->
    <div class="upload-card" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
      <div class="upload-icon">📷</div>
      <h3>Drag & drop your handwritten math photo here</h3>
      <p>Supports PNG, JPG, JPEG from phone cameras or scans</p>
      <button type="button" class="btn-sample" onclick="event.stopPropagation(); loadSamplePhoto()">⚡ Try With Sample Equation</button>
    </div>

    <!-- Loading State -->
    <div class="loading-state" id="loadingState">
      <div class="spinner"></div>
      <h3 style="margin-bottom: 6px;">Processing Handwritten Notes...</h3>
      <p style="color: var(--text-muted); font-size: 0.9rem;">Deskewing photo, segmenting lines, and generating LaTeX via beam search</p>
    </div>

    <!-- Results Section -->
    <div class="results-container" id="resultsSection">
      <div class="results-header">
        <div>
          <h2>Transcribed Mathematical Lines</h2>
          <p style="color: var(--text-muted); font-size: 0.9rem;" id="summaryText">Detected 0 lines</p>
        </div>
        <div class="actions-bar">
          <button class="btn btn-secondary" onclick="copyFullDocument()">📋 Copy LaTeX</button>
          <button class="btn btn-primary" onclick="downloadTexFile()">📥 Download .tex File</button>
        </div>
      </div>

      <div id="linesList"></div>
    </div>
  </div>

  <div class="toast" id="toast">Copied to clipboard!</div>

  <script>
    let currentLineData = [];

    // Drag and Drop
    const dropZone = document.getElementById('dropZone');
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
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

    async function loadSamplePhoto() {
      showLoading(true);
      try {
        const res = await fetch('/api/sample');
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert('Failed loading sample: ' + err);
      } finally {
        showLoading(false);
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
        alert('Failed to process photo: ' + err);
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
      document.getElementById('summaryText').textContent = `Detected ${currentLineData.length} mathematical equations`;
      const listEl = document.getElementById('linesList');
      listEl.innerHTML = '';

      currentLineData.forEach((line, idx) => {
        const card = document.createElement('div');
        card.className = 'line-card';
        card.innerHTML = `
          <div class="line-header">
            <span class="line-title">Equation #${idx + 1}</span>
            <div class="line-meta">
              <span class="tag tag-conf">${Math.round(line.confidence * 100)}% Confidence</span>
              <span class="tag ${line.compiles ? 'tag-syntax' : 'tag-warn'}">
                ${line.compiles ? '✓ Valid Syntax' : '⚠ Check Syntax'}
              </span>
            </div>
          </div>
          <div class="line-grid">
            <div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;">Cropped Line Photo</div>
              <div class="crop-preview-box">
                <img src="${line.crop_url}" alt="Equation Line ${idx + 1}">
              </div>
            </div>
            <div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;">Rendered Math Preview (KaTeX)</div>
              <div class="math-render-box" id="katex-${idx}"></div>
            </div>
          </div>
          <div class="latex-input-box">
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;">Editable LaTeX</div>
            <input type="text" class="latex-input" id="input-${idx}" value="${escapeHtml(line.latex)}" oninput="updateMath(${idx}, this.value)">
          </div>
        `;
        listEl.appendChild(card);
        renderKaTeX(idx, line.latex);
      });

      document.getElementById('resultsSection').style.display = 'block';
    }

    function renderKaTeX(idx, latex) {
      const el = document.getElementById(`katex-${idx}`);
      if (!el) return;
      try {
        katex.render(latex, el, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (e) {
        el.textContent = latex;
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
        showToast('Complete LaTeX copied to clipboard!');
      });
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.display = 'block';
      setTimeout(() => { t.style.display = 'none'; }, 2500);
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


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
            "crop_url": f"/static/crops/{crop_filename}",
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
    """Build a multi-line synthetic note image and return transcribed output."""
    sample_img_paths = list(Path("data/processed/train").glob("*.png"))[:3]
    if not sample_img_paths:
        return jsonify({"error": "No sample images available"}), 404

    # Stack 3 sample images vertically with spacing to simulate a full page note
    h_stack = []
    for p in sample_img_paths:
        img = cv2.imread(str(p))
        if img is not None:
            # Add top margin
            blank = np.full((35, img.shape[1], 3), 255, dtype=np.uint8)
            h_stack.append(blank)
            h_stack.append(img)

    # Pad widths to maximum
    max_w = max(item.shape[1] for item in h_stack)
    padded_stack = []
    for item in h_stack:
        w = item.shape[1]
        if w < max_w:
            pad = np.full((item.shape[0], max_w - w, 3), 255, dtype=np.uint8)
            item = np.hstack([item, pad])
        padded_stack.append(item)

    page_img = np.vstack(padded_stack)
    session_id = f"sample_{int(time.time())}"
    sample_path = UPLOAD_CACHE_DIR / f"{session_id}.png"
    cv2.imwrite(str(sample_path), page_img)

    pipeline = get_pipeline()
    tex_content, line_details, line_crops = pipeline.process_page(page_img)

    response_lines = []
    for detail, crop in zip(line_details, line_crops):
        crop_filename = f"{session_id}_crop_{crop.index}.png"
        crop_path = CROPS_CACHE_DIR / crop_filename
        cv2.imwrite(str(crop_path), crop.image)

        response_lines.append({
            "index": detail["index"],
            "crop_url": f"/static/crops/{crop_filename}",
            "latex": detail["latex"],
            "confidence": detail["confidence"],
            "compiles": detail["compiles"],
        })

    return jsonify({
        "session_id": session_id,
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
