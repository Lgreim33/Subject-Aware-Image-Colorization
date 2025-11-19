import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===

# Base directory is the folder where this script and coco_2015 live
BASE_DIR = Path(r"C:\Users\baamd9\Downloads\coco_2015")

# Only one split for you
SPLITS = ["test2015"]

TARGET_SIZE = 512  # final width and height = 512 x 512

# Folder names
ORIG_DIR_NAME = "images_original"
RESIZED_DIR_NAME = "images_512"
GRAY_DIR_NAME = "images_512_gray"


def resize_and_pad_image(img: Image.Image, target_size: int) -> Image.Image:
    """
    Resize an image so that the LONGER side becomes target_size,
    preserving aspect ratio; then pad the shorter side with black pixels
    to get a square (target_size x target_size).
    """
    w, h = img.size  # (width, height)

    if w == target_size and h == target_size:
        return img.copy()

    max_side = max(w, h)
    scale = target_size / max_side

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Create a black square canvas
    canvas = Image.new("RGB", (target_size, target_size), color=0)

    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2

    canvas.paste(img_resized, (left, top))

    return canvas


def process_split_resize(split: str):
    orig_dir = BASE_DIR / ORIG_DIR_NAME / split
    out_dir = BASE_DIR / RESIZED_DIR_NAME / split

    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(orig_dir) if f.lower().endswith(exts)]
    files = sorted(files)

    print(f"\nProcessing split: {split}")
    print(f"Found {len(files)} images in {orig_dir}")

    for fname in tqdm(files, desc=f"Resizing {split}"):
        in_path = orig_dir / fname
        out_path = out_dir / fname

        img = Image.open(in_path)
        img = img.convert("RGB")

        img_padded = resize_and_pad_image(img, TARGET_SIZE)
        img_padded.save(out_path)


def process_split_grayscale(split: str):
    resized_dir = BASE_DIR / RESIZED_DIR_NAME / split
    out_dir = BASE_DIR / GRAY_DIR_NAME / split

    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(resized_dir) if f.lower().endswith(exts)]
    files = sorted(files)

    print(f"\nProcessing grayscale for split: {split}")
    print(f"Found {len(files)} resized images in {resized_dir}")

    for fname in tqdm(files, desc=f"Grayscaling {split}"):
        in_path = resized_dir / fname
        out_path = out_dir / fname

        img = Image.open(in_path)
        img_gray = img.convert("L")  # 8-bit grayscale

        img_gray.save(out_path)


def main():
    # 1) Resize + pad
    for split in SPLITS:
        process_split_resize(split)

    # 2) Grayscale
    for split in SPLITS:
        process_split_grayscale(split)

    # 3) Simple count check
    print("\nSanity check (counts per split):")
    for split in SPLITS:
        orig_dir = BASE_DIR / ORIG_DIR_NAME / split
        resized_dir = BASE_DIR / RESIZED_DIR_NAME / split
        gray_dir = BASE_DIR / GRAY_DIR_NAME / split

        exts = (".jpg", ".jpeg", ".png")
        orig_n = len([f for f in os.listdir(orig_dir) if f.lower().endswith(exts)])
        resized_n = len([f for f in os.listdir(resized_dir) if f.lower().endswith(exts)])
        gray_n = len([f for f in os.listdir(gray_dir) if f.lower().endswith(exts)])

        print(f"{split}: original={orig_n}, resized={resized_n}, gray={gray_n}")


if __name__ == "__main__":
    main()
