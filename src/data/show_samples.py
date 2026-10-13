"""Display and export sample equation images with their ground truth LaTeX."""

import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def main():
    manifest_path = Path("data/processed/train_manifest.jsonl")
    if not manifest_path.exists():
        print("Manifest not found. Run strokes_to_image first.")
        return

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 4:
                break
            samples.append(json.loads(line))

    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 2.5 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for ax, item in zip(axes, samples):
        img_path = Path("data/processed") / item["image_path"]
        img = cv2.imread(str(img_path))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"LaTeX: {item['label']}", fontsize=11, pad=10)
        ax.axis("off")

    plt.tight_layout()
    out_preview = Path("data/processed/samples_preview.png")
    plt.savefig(out_preview, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample preview saved to {out_preview}")
    for item in samples:
        print(f"Image: {item['image_path']} | LaTeX: {item['label']}")


if __name__ == "__main__":
    main()
