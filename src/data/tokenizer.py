"""LaTeX Tokenizer and Vocabulary manager for Math2LaTeX.

Splits LaTeX expressions into atomic mathematical tokens:
- LaTeX commands (e.g., \\frac, \\sqrt, \\alpha)
- Escaped symbols (e.g., \\{, \\}, \\%)
- Operators and punctuation (e.g., +, -, =, ^, _, (, ), {, })
- Single digits and characters (e.g., 0-9, x, y, A-Z)

Includes normalization, encoding, decoding, and JSON persistence.
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Union


# Special token identifiers
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

# Regex pattern to match LaTeX tokens:
# 1. LaTeX command with letters: \\[a-zA-Z]+
# 2. LaTeX escaped symbol: \\[^a-zA-Z\s]
# 3. Any single non-whitespace character: \S
TOKEN_PATTERN = re.compile(r"(\\[a-zA-Z]+|\\[^a-zA-Z\s]|\S)")


def normalize_latex(text: str) -> str:
    """Normalize a LaTeX math string to canonical representation.
    
    Why:
        Reduces syntactic ambiguity (e.g., '\\le' vs '\\leq', extra spaces)
        so the neural network has fewer equivalent variations to learn.
    """
    if not text:
        return ""

    # Replace newlines and multi-spaces with single space
    s = re.sub(r"\s+", " ", text.strip())

    # Common canonical synonym replacements
    synonyms = [
        (r"\\le(?![a-zA-Z])", r"\\leq"),
        (r"\\ge(?![a-zA-Z])", r"\\geq"),
        (r"\\ne(?![a-zA-Z])", r"\\neq"),
        (r"\\to(?![a-zA-Z])", r"\\rightarrow"),
        (r"\\dots(?![a-zA-Z])", r"\\ldots"),
        (r"\\iff(?![a-zA-Z])", r"\\Longleftrightarrow"),
    ]
    for pattern, repl in synonyms:
        s = re.sub(pattern, repl, s)

    # Standardize space around structural brackets and operators
    # Remove unnecessary spaces inside { } or after ^ and _
    s = re.sub(r"\^\s+", "^", s)
    s = re.sub(r"_\s+", "_", s)
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)
    
    # Remove space after LaTeX commands when immediately followed by { or ^ or _
    s = re.sub(r"(\\[a-zA-Z]+)\s+([\{^_\(\[\)\]])", r"\1\2", s)

    # Normalize tokenization and join cleanly
    tokens = tokenize_latex(s)
    return detokenize_tokens(tokens)


def tokenize_latex(text: str) -> List[str]:
    """Tokenize a LaTeX string into atomic tokens."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text)


def detokenize_tokens(tokens: Sequence[str]) -> str:
    """Reconstruct a clean LaTeX string from tokens with correct spacing."""
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            # When a LaTeX command word (like \alpha, \log) is immediately followed
            # by an alphabetical letter or another command word, a space is mandatory.
            # Otherwise '\log' + 'x' would become '\logx' (undefined LaTeX command).
            if re.match(r"^\\[a-zA-Z]+$", prev) and re.match(r"^[a-zA-Z\\]", tok):
                out.append(" ")
        out.append(tok)
    return "".join(out)


class LatexTokenizer:
    """Manages token <-> ID mappings for LaTeX mathematical formulas."""

    def __init__(self, vocab_path: Optional[Union[str, Path]] = None):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # Initialize with special tokens
        for i, token in enumerate(SPECIAL_TOKENS):
            self.token2id[token] = i
            self.id2token[i] = token

        self.pad_id = self.token2id[PAD_TOKEN]
        self.start_id = self.token2id[START_TOKEN]
        self.end_id = self.token2id[END_TOKEN]
        self.unk_id = self.token2id[UNK_TOKEN]

        if vocab_path is not None:
            self.load(vocab_path)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def build_vocab(self, texts: Sequence[str], min_freq: int = 1) -> None:
        """Build vocabulary from a collection of LaTeX strings (training set only)."""
        counts: Dict[str, int] = {}
        for text in texts:
            norm = normalize_latex(text)
            for token in tokenize_latex(norm):
                counts[token] = counts.get(token, 0) + 1

        # Reset mappings to special tokens
        self.token2id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        self.id2token = {i: token for i, token in enumerate(SPECIAL_TOKENS)}

        # Add tokens meeting min_freq, sorted alphabetically for determinism
        sorted_tokens = sorted([t for t, c in counts.items() if c >= min_freq and t not in self.token2id])
        for tok in sorted_tokens:
            new_id = len(self.token2id)
            self.token2id[tok] = new_id
            self.id2token[new_id] = tok

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert LaTeX string to list of token IDs."""
        norm = normalize_latex(text)
        tokens = tokenize_latex(norm)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]

        if add_special_tokens:
            ids = [self.start_id] + ids + [self.end_id]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token IDs back into a LaTeX string."""
        tokens: List[str] = []
        for tid in token_ids:
            if tid == self.end_id and skip_special_tokens:
                break
            if skip_special_tokens and tid in (self.pad_id, self.start_id):
                continue
            token_str = self.id2token.get(tid, self.unk_token)
            tokens.append(token_str)
        return detokenize_tokens(tokens)

    @property
    def pad_token(self) -> str:
        return PAD_TOKEN

    @property
    def start_token(self) -> str:
        return START_TOKEN

    @property
    def end_token(self) -> str:
        return END_TOKEN

    @property
    def unk_token(self) -> str:
        return UNK_TOKEN

    def save(self, vocab_path: Union[str, Path]) -> None:
        """Save vocabulary mapping to a JSON file."""
        path = Path(vocab_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "token2id": self.token2id,
                "special_tokens": SPECIAL_TOKENS,
            }, f, indent=2, ensure_ascii=False)

    def load(self, vocab_path: Union[str, Path]) -> None:
        """Load vocabulary mapping from a JSON file."""
        path = Path(vocab_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self.pad_id = self.token2id[PAD_TOKEN]
        self.start_id = self.token2id[START_TOKEN]
        self.end_id = self.token2id[END_TOKEN]
        self.unk_id = self.token2id[UNK_TOKEN]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and save LaTeX tokenizer vocabulary")
    parser.add_argument("--train_manifest", default="data/processed/train_manifest.jsonl", help="Path to training manifest")
    parser.add_argument("--vocab_path", default="data/processed/vocab.json", help="Path to save vocabulary JSON")
    args = parser.parse_args()

    manifest_path = Path(args.train_manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        exit(1)

    labels = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            labels.append(data["label"])

    tokenizer = LatexTokenizer()
    tokenizer.build_vocab(labels)
    tokenizer.save(args.vocab_path)
    print(f"Built vocabulary with {tokenizer.vocab_size} tokens from {len(labels)} expressions.")
    print(f"Saved vocabulary to {args.vocab_path}")

    # Verify round-trip test on all labels
    all_passed = True
    for lbl in labels:
        norm = normalize_latex(lbl)
        encoded = tokenizer.encode(lbl)
        decoded = tokenizer.decode(encoded)
        if decoded != norm:
            print(f"[Roundtrip Mismatch] Original: {norm} | Decoded: {decoded}")
            all_passed = False
            break
    if all_passed:
        print("Roundtrip test passed on all manifest labels: decode(encode(x)) == normalize(x)!")
