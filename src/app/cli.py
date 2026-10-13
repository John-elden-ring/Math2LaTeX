"""Command Line Interface (CLI) for Math2LaTeX.

Converts a photo of handwritten math notes directly into a formatted LaTeX document.

Usage:
    python app/cli.py photo.jpg -o notes.tex
    python -m src.app.cli photo.jpg -o notes.tex
"""

import argparse
from pathlib import Path
import sys

from src.pipeline.page_to_tex import PageToTexPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Math2LaTeX CLI: Convert handwritten math photo notes into a LaTeX document."
    )
    parser.add_argument("photo", help="Path to input photo (.jpg, .png, etc.)")
    parser.add_argument(
        "-o", "--output",
        default="notes.tex",
        help="Output .tex filename (default: notes.tex)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest_model.pt",
        help="Path to trained model checkpoint (default: checkpoints/latest_model.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML configuration (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=4,
        help="Beam search size for decoding (default: 4)",
    )
    args = parser.parse_args()

    photo_path = Path(args.photo)
    if not photo_path.exists():
        print(f"[ERROR] Input photo not found: {photo_path}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    # Check fallback checkpoints if default does not exist
    if not ckpt_path.exists():
        alternatives = [
            Path("checkpoints/best_model.pt"),
            Path("checkpoints/overfit_100_model.pt"),
            Path("checkpoints/latest_model.pt"),
        ]
        found = False
        for alt in alternatives:
            if alt.exists():
                ckpt_path = alt
                found = True
                break
        if not found:
            print(f"[ERROR] No model checkpoint found at {args.checkpoint} or in checkpoints/.")
            print("Please run model training first: python -m src.model.train --overfit")
            sys.exit(1)

    print("=" * 60)
    print("Math2LaTeX Document Transcriber")
    print(f"Input Photo:   {photo_path}")
    print(f"Output File:   {args.output}")
    print(f"Model:         {ckpt_path}")
    print("=" * 60)

    try:
        pipeline = PageToTexPipeline(checkpoint_path=str(ckpt_path), config_path=args.config)
        tex_content, line_details, _ = pipeline.process_page(
            image_input=photo_path,
            output_tex_path=args.output,
            beam_size=args.beam_size,
        )

        print("\n" + "=" * 60)
        print(f"SUCCESS: Transcribed {len(line_details)} mathematical lines to '{args.output}'.")
        print("=" * 60)
        print("\nDocument Preview:")
        print("-" * 40)
        print(tex_content[:800] + ("\n..." if len(tex_content) > 800 else ""))
        print("-" * 40)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
