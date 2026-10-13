# Stage 6: Evaluation, Metrics, and Error Analysis

## Goal in Plain Words
Measure how well the model reads handwritten math when tested on completely unseen images, using fair mathematical metrics rather than guesswork. Automatically identify and save the hardest mistakes so we can continuously improve the system.

---

## The Three Core Metrics

### 1. Exact Match (%)
- **What it checks**: Did the model predict the *exact* sequence of LaTeX tokens correctly from start to finish?
- **Why it matters**: In mathematics, a single wrong digit or misplaced minus sign completely changes an equation ($e^{-x}$ vs $e^{x}$).
- **Challenge**: It is extremely strict. If a 50-token formula has 49 tokens correct and one missing comma, Exact Match awards 0%.

### 2. Normalized Levenshtein Edit Distance (0.0 to 1.0)
- **What it checks**: How many token/character insertions, deletions, or substitutions are needed to turn the model's prediction into the ground truth?
- **Formula**:
  $$\text{Normalized Edit Distance} = \frac{\text{Levenshtein}(pred, target)}{\max(\text{len}(pred), \text{len}(target))}$$
- **Score**:
  - `0.0`: Perfect prediction.
  - `0.05`: Tiny typo (e.g., predicted `x` instead of `y`).
  - `1.0`: Completely wrong or empty.
- **Why it is fair**: It rewards predictions that are 95% right, giving an honest reflection of model accuracy.

### 3. LaTeX Compile Rate (%)
- **What it checks**: Does the generated LaTeX compile without fatal syntax errors?
- **Checks applied**:
  - Balanced curly braces (`{` matches `}`).
  - Balanced `\left` and `\right` sizing delimiters.
  - No illegal dangling operators (like ending on `^` or `_` or `\`).
  - Validation via `pdflatex` or mathematical AST syntax checker.

---

## Worst 50 Mistakes Log (`reports/worst_50_mistakes.json`)
The evaluator sorts all test predictions by edit distance in descending order and exports the top 50 worst samples:
- Allows inspecting ambiguous handwriting (e.g., writer's `1` looks like `l` or `|`).
- Guides targeted data augmentation and real-world fine-tuning.
