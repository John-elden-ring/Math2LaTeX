# Math2LaTeX: Handwritten Math Notes to LaTeX

An end-to-end deep learning system that transcribes smartphone photographs of handwritten mathematical notes directly into compilable LaTeX code.

**Authors:**  
- **Adhik Puthenkattil**  
- **Goutham Krishna PS**  
- **Abhijith Jayan**  

**Institution:**  
*Department of Computer Science and Engineering*  
*Indian Institute of Information Technology (IIIT) Kottayam, Kerala, India*  

---

## Overview

- **Custom-Trained Architecture**: Deep visual ResNet-18 convolutional backbone + 2D sinusoidal spatial positional embeddings + 4-layer autoregressive Transformer sequence decoder.
- **Robust Real-World Preprocessing**: Automatic document deskewing, local adaptive Gaussian shadow removal, and smoothed horizontal projection profile line segmentation.
- **Syntactic Stabilization**: Automated delimiter balancing and orphan brace elimination to guarantee clean, compilable LaTeX output.
- **Interactive Web Interface**: Live browser application featuring side-by-side handwritten stroke preview, instant KaTeX visual formula rendering, token-level confidence scoring, and 1-click `.tex` project export.

---

## Project Structure

```text
math2latex/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── base.yaml              # Hyperparameters & model configurations
├── data/
│   ├── processed/
│   │   └── vocab.json         # 1024-token LaTeX vocabulary
│   └── sample_gallery.json    # Verified benchmark sample gallery
├── src/
│   ├── app/
│   │   ├── web.py             # Interactive Flask web application with KaTeX preview
│   │   └── cli.py             # CLI for single-image and batch transcription
│   ├── model/
│   │   ├── img2latex.py       # Full sequence-to-sequence model definition
│   │   ├── encoder.py         # ResNet-18 visual feature grid + 2D positional encoding
│   │   ├── decoder.py         # Autoregressive Transformer decoder
│   │   ├── infer.py           # Beam search inference pipeline
│   │   ├── train.py           # Training loop with warmup, fp16 & gradient clipping
│   │   └── evaluate.py        # Token error rate, BLEU, and edit distance evaluation
│   ├── pipeline/
│   │   ├── preprocess.py      # Deskewing, shadow removal & Sauvola binarization
│   │   ├── split_lines.py     # Projection profile line segmentation
│   │   └── page_to_tex.py     # End-to-end page image to .tex transcription
│   └── data/
│       ├── tokenizer.py       # LaTeX tokenizer and vocabulary manager
│       ├── dataset.py         # PyTorch Dataset and batch collator
│       ├── augment.py         # Realistic camera/ink/lighting augmentations
│       ├── strokes_to_image.py# Vector stroke rasterizer
│       └── download.py        # Dataset downloader
├── notebooks/                 # Analysis and training demonstration notebooks
└── docs/                      # Architectural stage documentation (stage_1 to stage_10)
```

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/John-elden-ring/Math2LaTeX.git
cd Math2LaTeX
pip install -r requirements.txt
```

### 2. Model Checkpoint Setup

Place your trained model checkpoint at:
```text
checkpoints/best_model.pt
```
*(Model weights can be produced via `python -m src.model.train` or downloaded from releases).*

### 3. Launch the Interactive Web Application

```bash
python -m src.app.web
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```
- Upload any mobile phone photograph of handwritten math.
- Or select from the preloaded benchmark test cases in the gallery to test live inference instantly.

### 4. Transcribe via CLI

Transcribe a single handwritten equation image or full notebook page:
```bash
python -m src.app.cli path/to/equation.png -o output.tex
```

---

## Architectural Stages

Detailed implementation breakdowns for each stage are available in `docs/`:
- [Stage 1: Configuration & Setup](docs/stage_1.md)
- [Stage 2: Stroke Rendering & Tokenization](docs/stage_2.md)
- [Stage 3: ResNet-18 & 2D Positional Embeddings](docs/stage_3.md)
- [Stage 4: Overfit Sanity Milestone](docs/stage_4.md)
- [Stage 5: Scaled Training & Checkpoints](docs/stage_5.md)
- [Stage 6: Evaluation & Metrics](docs/stage_6.md)
- [Stage 7: Beam Search Decoding](docs/stage_7.md)
- [Stage 8: Page Deskew & Line Segmentation](docs/stage_8.md)
- [Stage 9: CLI and Web Deployment](docs/stage_9.md)
- [Stage 10: Notebook Domain Adaptation](docs/stage_10.md)

---

## License
MIT License. Developed for research and educational purposes at IIIT Kottayam.
