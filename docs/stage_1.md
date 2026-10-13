# Stage 1: Setup and Configuration

## Goal in Plain Words
Set up a clean, well-organized home for the project. Make sure all library dependencies are defined, all settings are in one single configuration file (`configs/base.yaml`), and any part of the project can read those settings without hardcoding values in code.

---

## Why We Built It This Way

1. **One Central Config (`configs/base.yaml`)**:
   - Rather than sprinkling magic numbers (like image height = 128, learning rate = 0.0001, beam size = 4) across multiple files, every single knob is kept in `configs/base.yaml`.
   - If we want to experiment with a larger batch size or different dropout rate, we change one line in the YAML file.

2. **Modular File Layout**:
   - `src/data/` only cares about getting, drawing, tokenizing, and feeding data.
   - `src/model/` only cares about neural network layers, training, loss, and weights.
   - `src/pipeline/` only cares about whole camera photos (fixing tilt, finding lines).
   - `src/app/` only cares about user interfaces (command line and web page).
   - `tests/` allows running automated checks so changes don't break earlier work.

3. **Plain Code, No Hidden Magic**:
   - We use standard PyTorch, torchvision, and NumPy.
   - Code comments explain *why* something is done (for example, why height is locked to 128 while width varies).

---

## Key Settings at a Glance
- **Image Height**: `128` pixels (preserves stroke details like dots and exponents).
- **Model Dimension**: `256` (compact enough to train quickly, expressive enough for complex math).
- **Decoder**: 4 Transformer layers with 8 attention heads.
- **Max Sequence Length**: 256 LaTeX tokens.
- **Batch Size**: 32.
- **Learning Rate**: `0.0001` (1e-4) with AdamW.
