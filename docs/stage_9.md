# Stage 9: CLI and Web Application

## Goal in Plain Words
Give users two intuitive ways to use the system:
1. A fast terminal tool for batch processing notes.
2. A sleek, interactive web page with drag-and-drop, line-by-line inspection, live math rendering, and `.tex` export.

---

## 1. Command Line Interface (`app/cli.py`)
Run from terminal:
```bash
python app/cli.py my_notes.jpg -o output.tex
```
- Automatically deskews the photo, crops each math line, runs beam search, compiles individual lines, and joins them into an `amsmath` document.
- Prints confidence scores and highlights syntax warnings.

---

## 2. Interactive Web Application (`app/web.py`)
Run from terminal:
```bash
python app/web.py
```
Visit `http://localhost:5000` in your web browser.

### Features
- **Drag-and-Drop Upload**: Drop any phone photo or scan directly onto the browser window.
- **One-Click Sample Test**: Try the pipeline instantly with built-in handwritten notes without needing to upload your own file.
- **Side-by-Side Verification**: View the cropped camera image alongside the transcribed LaTeX code.
- **Live KaTeX Rendering**: Equations render into beautiful mathematical typography in real-time as you type or adjust tokens.
- **Direct Editing & Export**: Edit any equation line if a symbol was misread, and click **Download .tex File** or **Copy LaTeX** with a single click.
