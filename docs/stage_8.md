# Stage 8: Page Preprocessing and Line Segmentation

## Goal in Plain Words
A user takes a phone picture of an entire page of notes. The page might be tilted, have shadows from their hand or phone, and contain multiple lines of math. We must straighten the page, erase the shadows, cut each line of math out cleanly, and send each line to the model.

---

## The 4 Steps in the Pipeline

### 1. Page Deskewing (`src/pipeline/preprocess.py`)
- Detects the orientation of text on the page using OpenCV contour bounding geometry.
- Automatically rotates the image by $-\theta$ (up to $\pm 15^\circ$) so lines run horizontally across the page.

### 2. Shadow Removal & Binarization (`clean_and_binarize`)
- Overhead lighting and phone shadows cause dark patches across paper notes. Standard global thresholding fails (makes shadowed areas completely black).
- We use **Gaussian Adaptive Thresholding**:
  - Analyzes a local $25 \times 25$ pixel window around every single pixel.
  - Dynamically calculates the threshold based on the local neighborhood mean.
  - Ink turns crisp black; shadowy paper becomes pure white.

### 3. Horizontal Projection Profile (`src/pipeline/split_lines.py`)
- We count the total number of ink pixels in every single horizontal row:
  $$\text{Ink}(y) = \sum_{x=0}^{W-1} \mathbb{I}(\text{pixel}(y, x) == \text{ink})$$
- When plotted, lines of equations look like hills (high ink count), and blank spaces between lines look like deep valleys (zero ink).
- Finding the valleys gives the exact vertical boundary $(y_1, y_2)$ of each equation.
- We smooth the projection profile with a 1D filter so small letter dots don't accidentally split a single equation into two pieces.
- Lines separated by less than `min_line_gap` (15 pixels) are safely merged.

### 4. Aspect-Ratio Preserving Normalization
- Each cropped equation is resized to fixed height **128 pixels**, preserving its natural width.
- Padded with clean background so the encoder sees standardized input.
