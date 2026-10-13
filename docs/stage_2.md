# Stage 2: Data Pipeline, Rendering, and Tokenization

## Goal in Plain Words
Take handwritten math notes recorded as pen coordinates (strokes in InkML files), turn them into clean pictures of equations, and teach the computer how to read the LaTeX code token by token.

---

## What Each File Does

1. **`src/data/download.py`**:
   - Downloads the Google MathWriting dataset archive directly from official storage.
   - Extracts the XML/InkML files cleanly into `data/raw/`.

2. **`src/data/strokes_to_image.py`**:
   - Parses the `<trace>` tags (pen movements) and `<annotation type="normalizedLabel">` (LaTeX).
   - Draws each stroke as dark ink on a white page.
   - Resizes every equation line to a fixed height of **128 pixels**, keeping the width proportional.
   - Adds slight random variation to pen width (stroke jitter) so the model doesn't overfit to one pen style.
   - Writes `train_manifest.jsonl`, `valid_manifest.jsonl`, and `test_manifest.jsonl`.
   - **Verification rule**: Checks mathematically that `train`, `val`, and `test` have zero overlapping IDs.

3. **`src/data/tokenizer.py`**:
   - Splits LaTeX into atomic pieces:
     - Math commands: `\frac`, `\sqrt`, `\alpha`, `\sum`
     - Operators & brackets: `+`, `-`, `=`, `^`, `_`, `{`, `}`
     - Single characters & numbers: `x`, `y`, `2`, `n`
   - Special tokens: `<pad>` (0), `<start>` (1), `<end>` (2), `<unk>` (3).
   - Normalizes whitespace and canonical synonyms (`\le` -> `\leq`, `\to` -> `\rightarrow`).
   - Ensures lossless roundtrip: `decode(encode(normalize(x))) == normalize(x)`.

4. **`src/data/augment.py`**:
   - Simulates real camera photos of handwritten notes during training:
     - Paper color, grain, and cream tone
     - Uneven lighting and phone camera shadow gradients
     - Faint notebook ruled lines
     - Black ink and blue ballpoint ink variations
     - Line dilation/erosion (thin vs. thick pen)
     - Slight rotation (up to 3 degrees), camera blur, and JPEG compression
   - Augmentation is **strictly disabled** during validation and testing.

5. **`src/data/dataset.py`**:
   - PyTorch `MathDataset` and custom dynamic batch collator `math_collate_fn`.
   - Batches pad image widths to the widest image in that specific batch (fixed height 128px).
   - Batches pad token sequences to the longest sequence in that batch with `<pad>`.
