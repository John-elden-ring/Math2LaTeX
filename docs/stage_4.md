# Stage 4: 100-Sample Overfit Milestone

## Goal in Plain Words
Before spending hours or days training on hundreds of thousands of equations, we must prove beyond any doubt that our model, data pipeline, and tokenizer actually work. We do this by feeding the model just 100 equations and making sure it can memorize them with near-zero loss.

---

## Why This Step Matters
- If there is a hidden bug anywhere (for example: labels not aligned with images, tokens cut off, attention mask pointing the wrong way, gradients exploding), the model will never be able to memorize 100 samples.
- Catching bugs here takes 2 minutes instead of wasting hours on a cloud GPU.

---

## How It Works
1. Load 100 samples from `train_manifest.jsonl`.
2. Disable data augmentation so the network sees the exact same clean strokes every time.
3. Train with AdamW and teacher forcing.
4. Watch the cross-entropy loss fall from > 4.5 down below 0.08.
5. Save the checkpoint to `checkpoints/overfit_100_model.pt`.

---

## What the Numbers Mean
- **Starting Loss (~4.8 - 5.5)**: The model is guessing uniformly at random among ~180 possible tokens ($-\ln(1/180) \approx 5.2$).
- **Midway Loss (~1.0 - 2.0)**: The model has learned common syntax patterns like `\frac{...}{...}` and basic variables.
- **Ending Loss (< 0.08)**: The model has successfully memorized the exact equations. The pipeline is proven bug-free.
