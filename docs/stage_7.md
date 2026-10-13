# Stage 7: Inference with Beam Search

## Goal in Plain Words
When transcribing an equation, make sure the model does not pick a short-sighted wrong token that derails the entire formula. Use Beam Search to explore multiple promising candidate transcriptions simultaneously.

---

## Greedy Search vs. Beam Search

### Greedy Search (Beam Size = 1)
- At each step, the model picks whichever token has the single highest probability.
- **The Problem**: If the model is choosing between `+` (51%) and `-` (49%), choosing `+` might lead down a dead end 5 tokens later where no valid math exists. Once a wrong token is chosen, the model cannot backtrack.

### Beam Search (Beam Size = 4 or 5)
- Instead of keeping only 1 candidate, the model keeps the **top $K$ highest-scoring sequences** (beams) at each step.
- At step 1: Keeps top 4 initial tokens.
- At step 2: Expands each of the 4 beams into all possible next tokens (4 * $V$ options), scores all combinations, and retains the top 4 combined sequences.
- Continues until the candidates emit the `<end>` token.
- **Length Penalty**: Longer equations naturally accumulate lower joint probabilities. We apply Wu et al. length normalization:
  $$\text{Score} = \frac{\log P(\text{tokens})}{((5 + \text{len}) / 6)^\alpha}$$
- **Confidence Score**: The model calculates the geometric mean probability of the chosen sequence, giving a clear percentage confidence (e.g. 96.4%).
