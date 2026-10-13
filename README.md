# Math2LaTeX: Handwritten Math Notes to LaTeX

An end-to-end, custom-trained deep learning system that transforms photographs of handwritten mathematical notes into clean, compilable LaTeX documents.

Built strictly according to the design specification:
- **Trained from scratch**: No ready-made math OCR APIs or models (e.g., pix2tex, Nougat, Texify, Mathpix).
- **Architecture**: Pretrained ResNet-18 visual feature extractor + 2D Positional Embeddings + 4-layer Transformer sequence decoder.
- **Robust to real-world camera photos**: Realistic paper/ink/lighting augmentations and full-page segmentation pipeline.
- **Explainable**: Simple, modular PyTorch code accompanied by clear documentation in `docs/stage_N.md`.

---

## Project Structure

```text
math2latex/
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml              # All project settings & hyperparameters
├── data/
│   ├── raw/                   # Raw downloaded InkML archives (not in git)
│   ├── processed/             # Rasterized images & vocab.json (not in git)
│   └── real/                  # Real phone photo samples for fine-tuning
├── src/
│   ├── data/
│   │   ├── download.py        # Fetch MathWriting & CROHME datasets
│   │   ├── strokes_to_image.py# Render vector strokes into 128px grayscale images
│   │   ├── tokenizer.py       # LaTeX tokenizer & vocabulary manager
│   │   ├── dataset.py         # PyTorch Dataset, batch collator & DataLoader
│   │   └── augment.py         # Realistic photo/ink/camera augmentations
│   ├── model/
│   │   ├── encoder.py         # ResNet-18 feature grid + 2D positional encoding
│   │   ├── decoder.py         # Transformer autoregressive decoder
│   │   ├── img2latex.py       # Full sequence-to-sequence model
│   │   ├── train.py           # Training loop with warmup, fp16 & gradient clipping
│   │   ├── evaluate.py        # Exact match, normalized edit distance & compile rate
│   │   └── infer.py           # Single-image inference with beam search
│   ├── pipeline/
│   │   ├── preprocess.py      # Page straightening, shadow removal & binarization
│   │   ├── split_lines.py     # Projection-based handwritten line segmentation
│   │   └── page_to_tex.py     # End-to-end page photo -> complete .tex document
│   └── app/
│       ├── cli.py             # Command-line interface
│       └── web.py             # Interactive web application with live LaTeX preview
├── tests/                     # Unit tests for tokenizer, dataset, model & pipeline
├── notebooks/                 # Analysis and visualization notebooks
└── docs/                      # Stage-by-stage explainable notes (stage_1 to stage_10)
```

---

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Verify Data Pipeline & Model Sanity Checks

Run automated test suite:
```bash
pytest tests/ -v
```

### 3. Run the Overfit Sanity Check (Stage 4)

```bash
python -m src.model.train --config configs/base.yaml --overfit
```

### 4. Convert a Page Photo via CLI (Stage 9)

```bash
python -m src.app.cli path/to/photo.jpg -o notes.tex
```

### 5. Launch the Web App (Stage 9)

```bash
python -m src.app.web
```
Open `http://localhost:5000` in your browser.

---

## Dataset Statistics

- **Google MathWriting**: ~230,000 human-written samples + ~360,000 synthetic samples (total ~590,000 samples).
- **MathWriting Excerpt**: ~3,000 samples for immediate local development and quick tests.
- **CROHME**: Benchmark ink stroke expressions.
- **Real Photo Set**: At least 200 real notebook line photos in `data/real/` (50 reserved strictly for blind testing, remainder for fine-tuning).

---

## Documentation

Each development stage is explained simply in the `docs/` folder:
- [Stage 1: Setup & Configuration](docs/stage_1.md)
- [Stage 2: Data Pipeline, Rendering & Tokenization](docs/stage_2.md)
- [Stage 3: Model Architecture (ResNet + Transformer)](docs/stage_3.md)
- [Stage 4: 100-Sample Overfit Milestone](docs/stage_4.md)
- [Stage 5: Scaled Training Run & Checkpointing](docs/stage_5.md)
- [Stage 6: Evaluation, Metrics & Error Analysis](docs/stage_6.md)
- [Stage 7: Beam Search Inference](docs/stage_7.md)
- [Stage 8: Page Preprocessing & Line Segmentation](docs/stage_8.md)
- [Stage 9: CLI and Web Application](docs/stage_9.md)
- [Stage 10: Scaling Up & Real-World Photo Fine-Tuning](docs/stage_10.md)
