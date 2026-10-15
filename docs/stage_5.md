# Stage 5: Scaled Training Run & Checkpointing

## Goal in Plain Words
Scale up from the 100-sample test to full training on 50,000+ equations with data augmentation turned on. Make sure training can pause and resume safely if a free Google Colab or Kaggle session disconnects.

---

## Key Training Techniques Explained

1. **AdamW Optimizer & Weight Decay**:
   - Updates weights based on adaptive learning rates.
   - Weight decay (0.01) prevents the network weights from growing excessively large, acting as a regularizer against overfitting.

2. **Learning Rate Warmup & Cosine Decay**:
   - **Warmup**: In the first 2 epochs, learning rate slowly increases from 0 to `1e-4`. This prevents sudden large gradient updates before the Transformer attention weights stabilize.
   - **Cosine Decay**: Gradually tapers the learning rate down towards `1e-6` so the model settles gently into an optimal minimum.

3. **Gradient Clipping (`max_norm = 1.0`)**:
   - Long sequence decoders can occasionally suffer from gradient explosions when calculating cross-attention over many tokens.
   - Clipping normalizes gradients so their magnitude never exceeds 1.0.

4. **Mixed Precision (FP16)**:
   - When running on a GPU (CUDA), PyTorch uses `torch.amp.autocast('cuda')`.
   - Speeds up matrix multiplication by 2x to 3x and cuts VRAM usage in half without losing numerical accuracy.

5. **Bulletproof Checkpointing**:
   - Every epoch is saved as `checkpoints/latest_model.pt` containing the model weights, optimizer state, scheduler state, and epoch number.
   - If training pauses or times out, training resumes instantly via `--resume checkpoints/latest_model.pt`.
   - The validation-best weights are preserved in `checkpoints/best_model.pt`.

---

## Empirical Local GPU Benchmark (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Dataset Size**: 10,500 expressions (Google MathWriting real ink + synthetic math augmentation)
- **Vocabulary Size**: 223 atomic LaTeX tokens
- **Epochs Completed**: 20
- **Batch Size**: 32 (329 batches/epoch, mixed precision `fp16`)
- **Starting Loss**: 5.6167
- **Final Training Loss**: 0.2076
- **Validation Compilation Rate**: 96.00%
- **Validation Mean Edit Distance**: 0.1818
- **Checkpoint Artifacts**: `checkpoints/best_model.pt` (188.7 MB), `checkpoints/latest_model.pt` (188.7 MB)
