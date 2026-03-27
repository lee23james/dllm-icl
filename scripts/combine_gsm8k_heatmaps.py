import os
from pathlib import Path

from PIL import Image


def combine_side_by_side(
    left_path: str,
    right_path: str,
    out_path: str,
) -> None:
    """
    Combine two heatmap PNGs horizontally into a single figure.

    Left:  gsm8k_0_avg (interpreted as query at end / Pos=4)
    Right: gsm8k_4_avg (interpreted as query at beginning / Pos=0)
    """
    left_img = Image.open(left_path)
    right_img = Image.open(right_path)

    # Resize to same height (keep aspect ratio)
    target_h = min(left_img.height, right_img.height)

    def resize_to_height(img: Image.Image, h: int) -> Image.Image:
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)

    left_resized = resize_to_height(left_img, target_h)
    right_resized = resize_to_height(right_img, target_h)

    total_w = left_resized.width + right_resized.width
    combined = Image.new("RGB", (total_w, target_h), "white")
    combined.paste(left_resized, (0, 0))
    combined.paste(right_resized, (left_resized.width, 0))

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"Saved combined figure to: {out_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    left_path = project_root / "heatmap_results" / "gsm8k_0_avg" / "gsm8k.png"
    right_path = project_root / "heatmap_results" / "gsm8k_4_avg" / "gsm8k.png"
    out_path = project_root / "heatmap_results" / "gsm8k_pos4_vs_pos0.png"

    if not left_path.exists():
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right image not found: {right_path}")

    combine_side_by_side(str(left_path), str(right_path), str(out_path))


if __name__ == "__main__":
    main()

