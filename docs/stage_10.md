# Stage 10: Scaling Up & Real-World Photo Fine-Tuning

## Goal in Plain Words
Computer-drawn pen strokes from touchscreens are too clean compared to genuine phone camera photos taken in bad lighting on physical lined paper. We must scale up training to 500,000+ samples, fine-tune the model directly on Adhik's real notebook photos, and report separate performance scores for synthetic strokes vs. real camera photos.

---

## 1. The Large-Scale Run (500,000+ Equations)
- **MathWriting Full Dataset**:
  - ~230,000 human-written touchscreen samples
  - ~360,000 synthetic multi-symbol expressions
  - Total: ~590,000 samples
- **Execution on Free Cloud GPUs (Kaggle / Colab)**:
  - Single P100 / T4 GPU with mixed precision (fp16) takes ~3-4 hours per full pass.
  - Multi-worker PyTorch `DataLoader` (`num_workers=4`, `pin_memory=True`) keeps the GPU feed saturated.
  - Automatic resumption ensures zero progress is lost if the cloud session disconnects.

---

## 2. Real-World Photo Dataset (`data/real/`)
- **Protocol**:
  - Adhik photographs physical lined/grid notebook notes containing at least 200 distinct equation lines.
  - Bounding boxes and ground-truth LaTeX annotations are placed in `data/real/real_notes_manifest.jsonl`.
- **Strict Split Rule**:
  - **50 lines** are reserved *strictly and permanently* as the blind `real_test` set.
  - **150+ lines** are used for fine-tuning.
  - **Zero data leakage**: `test_ids` are never shown during training or tuning.

---

## 3. Real Photo Fine-Tuning Strategy
If you train *only* on the 150 real photos, a neural network suffers from **catastrophic forgetting** (it memorizes the 150 handwriting samples but forgets rare math symbols like `\oint`, `\nabla`, Greek letters).
- **Solution — Balanced Replay Mixing**:
  - Every batch during fine-tuning contains **50% real camera photos** and **50% photo-augmented MathWriting equations**.
  - Small learning rate ($2 \times 10^{-5}$) with Cosine decay.
  - Runs for 5 to 10 passes.

---

## 4. Reporting Real Results Separately (Mandatory Rule)
Evaluation reports must display two separate columns:
| Metric | Drawn Test Set (MathWriting) | Real Photo Test Set (`data/real`) |
| :--- | :---: | :---: |
| **Exact Match** | e.g. 78.4% | e.g. 64.2% |
| **Normalized Edit Distance** | e.g. 0.048 | e.g. 0.091 |
| **LaTeX Compile Rate** | e.g. 98.2% | e.g. 94.6% |

> The tool counts as working only when the real photo numbers are strong. We never claim success from the drawn test set alone.
