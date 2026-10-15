"""Single-image inference with Beam Search for Math2LaTeX.

Implements:
- Beam search decoding (configurable beam size 4 or 5)
- Length penalty normalization
- Output confidence score estimation
- Conversion from image file to predicted LaTeX string
"""

import argparse
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from src.config import load_config
from src.data.tokenizer import LatexTokenizer, normalize_latex
from src.model.img2latex import Img2Latex


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Hypothesis:
    """Represents an active search candidate hypothesis in beam search."""
    def __init__(self, tokens: List[int], log_prob: float):
        self.tokens = tokens
        self.log_prob = log_prob

    @property
    def length(self) -> int:
        return len(self.tokens)

    def score(self, alpha: float = 0.7) -> float:
        """Length-normalized score using Wu et al. length penalty."""
        # Penalty: ((5 + len) / 6) ^ alpha
        penalty = ((5.0 + self.length) / 6.0) ** alpha
        return self.log_prob / penalty


@torch.no_grad()
def beam_search_decode(
    model: Img2Latex,
    image_tensor: torch.Tensor,
    beam_size: int = 4,
    max_tokens: int = 256,
    length_penalty_alpha: float = 0.7,
) -> Tuple[List[int], float]:
    """Perform beam search decoding on a single image.
    
    Args:
        model: Img2Latex model instance.
        image_tensor: Normalized image tensor of shape (1, 3, 128, W).
        beam_size: Beam width (e.g. 4 or 5).
        max_tokens: Maximum tokens to generate.
        length_penalty_alpha: Length normalization exponent.
        
    Returns:
        best_tokens: List of token IDs (excluding <start> and <end>).
        confidence: Estimated confidence score between 0.0 and 1.0.
    """
    model.eval()
    device = image_tensor.device

    # Step 1: Compute encoder feature map once: shape (1, S_enc, model_dim)
    memory, _ = model.encoder(image_tensor)

    # Initial beam: starting with <start> token
    beams: List[Hypothesis] = [Hypothesis(tokens=[model.start_id], log_prob=0.0)]
    completed: List[Hypothesis] = []

    for step in range(max_tokens - 1):
        candidates: List[Hypothesis] = []

        for hyp in beams:
            # If hypothesis already ended, retain it
            if hyp.tokens[-1] == model.end_id:
                candidates.append(hyp)
                continue

            # Prepare decoder input tensor: (1, seq_len)
            tgt_tensor = torch.tensor([hyp.tokens], dtype=torch.long, device=device)
            logits = model.decoder(tgt_tensor, memory)  # (1, seq_len, vocab_size)

            # Log probabilities of next token
            next_logits = logits[0, -1, :]  # (vocab_size,)
            log_probs = F.log_softmax(next_logits, dim=-1)

            # Top beam_size candidates for this hypothesis
            top_log_probs, top_indices = torch.topk(log_probs, k=beam_size)

            for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_tokens = hyp.tokens + [idx]
                new_lp = hyp.log_prob + lp
                candidates.append(Hypothesis(tokens=new_tokens, log_prob=new_lp))

        # Sort all candidates by length-penalized score descending
        candidates.sort(key=lambda h: h.score(length_penalty_alpha), reverse=True)

        # Select top beam_size hypotheses
        beams = candidates[:beam_size]

        # Check if all top hypotheses have finished
        if all(h.tokens[-1] == model.end_id for h in beams):
            completed = beams
            break

    if not completed:
        completed = beams

    # Pick best overall hypothesis
    best_hyp = max(completed, key=lambda h: h.score(length_penalty_alpha))

    # Strip <start> and <end> tokens
    clean_tokens = [t for t in best_hyp.tokens if t not in (model.start_id, model.end_id, model.pad_id)]

    # Calculate average probability as confidence
    avg_log_prob = best_hyp.log_prob / max(1, len(best_hyp.tokens) - 1)
    confidence = float(np.exp(np.clip(avg_log_prob, -10.0, 0.0)))

    return clean_tokens, confidence


def sanitize_latex(s: str) -> str:
    """Auto-close dangling environments and balance unclosed braces."""
    if not s:
        return ""
    s = s.strip()
    if s.endswith("\\"):
        s = s[:-1].strip()

    # Balance and auto-close unclosed environments (e.g., \begin{matrix} -> \end{matrix})
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([a-zA-Z*]+)\}", s):
        cmd, env = m.group(1), m.group(2)
        if cmd == "begin":
            stack.append(env)
        elif cmd == "end":
            if stack and stack[-1] == env:
                stack.pop()
    while stack:
        env = stack.pop()
        s += f"\\end{{{env}}}"

    # Strip orphan closing curly braces and balance unclosed braces
    clean_chars = []
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
            clean_chars.append(ch)
        elif ch == "}":
            if depth > 0:
                depth -= 1
                clean_chars.append(ch)
            # Skip orphan '}' when depth == 0 to prevent KaTeX fatal syntax error
        else:
            clean_chars.append(ch)
    if depth > 0:
        clean_chars.extend(["}"] * depth)
    s = "".join(clean_chars)

    # Balance \left and \right delimiters (prevent KaTeX Expected \right error)
    num_left = len(re.findall(r"\\left(?![a-zA-Z])", s))
    num_right = len(re.findall(r"\\right(?![a-zA-Z])", s))
    if num_left > num_right:
        diff = num_left - num_right
        for _ in range(diff):
            bare_close = [m.start() for m in re.finditer(r"(?<!\\right)\)", s)]
            if bare_close:
                # Promote outermost un-prefixed ')' to '\right)'
                idx = bare_close[-1]
                s = s[:idx] + "\\right)" + s[idx + 1:]
            else:
                s += "\\right."
    elif num_right > num_left:
        diff = num_right - num_left
        for _ in range(diff):
            s = re.sub(r"\\right(?=[)\]\}|.])", "", s, count=1)

    return s


class LatexPredictor:
    """End-to-end wrapper for predicting LaTeX from single image crops."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        config_path: str = "configs/base.yaml",
        device: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load Tokenizer
        vocab_path = self.config["data"]["vocab_path"]
        self.tokenizer = LatexTokenizer(vocab_path)

        # Initialize Model
        self.model = Img2Latex(
            vocab_size=self.tokenizer.vocab_size,
            model_dim=self.config["model"]["model_dim"],
            decoder_layers=self.config["model"]["decoder_layers"],
            heads=self.config["model"]["heads"],
            feedforward_dim=self.config["model"]["feedforward_dim"],
            dropout=self.config["model"]["dropout"],
            max_tokens=self.config["model"]["max_tokens"],
            pad_id=self.tokenizer.pad_id,
            start_id=self.tokenizer.start_id,
            end_id=self.tokenizer.end_id,
            pretrained_backbone=False,
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def preprocess_image(self, img_input: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """Preprocess input image array or path to fixed height 128px tensor."""
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input))
            if img is None:
                raise ValueError(f"Could not read image from {img_input}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if img_input.ndim == 2:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 3:
                img_rgb = img_input
            else:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        target_h = 128

        # Resize keeping aspect ratio
        scale = target_h / float(h)
        new_w = max(int(round(w * scale)), 32)
        resized = cv2.resize(img_rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # Tensor conversion and normalization
        tensor = self.transform(resized).unsqueeze(0).to(self.device)  # (1, 3, 128, new_w)
        return tensor

    def predict(
        self,
        img_input: Union[str, Path, np.ndarray],
        beam_size: int = 4,
    ) -> Tuple[str, float]:
        """Predict LaTeX and return confidence score."""
        tensor = self.preprocess_image(img_input)
        token_ids, conf = beam_search_decode(
            model=self.model,
            image_tensor=tensor,
            beam_size=beam_size,
            max_tokens=self.config["model"]["max_tokens"],
        )
        latex_str = self.tokenizer.decode(token_ids)
        latex_str = sanitize_latex(latex_str)
        return latex_str, conf


def main():
    parser = argparse.ArgumentParser(description="Infer LaTeX from single equation image")
    parser.add_argument("image", help="Path to input equation image")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam search width")
    args = parser.parse_args()

    predictor = LatexPredictor(checkpoint_path=args.checkpoint, config_path=args.config)
    latex, conf = predictor.predict(args.image, beam_size=args.beam_size)
    print("\n" + "=" * 40)
    print(f"Predicted LaTeX: {latex}")
    print(f"Confidence:     {conf:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()
