import json
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Decode and overlay masks for all inference outputs.")
    parser.add_argument("--json_dir",  required=True, help="Folder containing inference .json files (e.g. GL123_345.json)")
    parser.add_argument("--img_dir",   required=True, help="Folder containing original .png images (e.g. GL123_345.png)")
    parser.add_argument("--output_dir",required=True, help="Folder to save overlay images")
    parser.add_argument("--alpha",     type=float, default=0.5, help="Mask overlay transparency (default: 0.5)")
    return parser.parse_args()

def process_single(json_path, img_path, output_dir, alpha):
    stem = os.path.splitext(os.path.basename(json_path))[0]  # e.g. 'GL123_345'

    # Load JSON
    with open(json_path) as f:
        data = json.load(f)

    # Load original image
    img_array = np.array(Image.open(img_path).convert("RGB"))

    if not data.get("pred_masks"):
        print(f"  [SKIP] No masks in {json_path}")
        return

    # Overlay each mask and save as a separate file
    for i, pred_mask in enumerate(data["pred_masks"]):
        rle = pred_mask
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")

        mask_array = maskUtils.decode(rle)  # shape: (H, W)

        red_mask = np.zeros_like(img_array)
        red_mask[..., 0] = 255  # Red channel only
        overlay = (
            img_array * (1 - alpha) + red_mask * mask_array[..., None] * alpha
        ).astype(np.uint8)

        # e.g. GL123_345_mask0.png, GL123_345_mask1.png ...
        out_filename = f"{stem}_mask{i}.png"
        out_path = os.path.join(output_dir, out_filename)
        Image.fromarray(overlay).save(out_path)
        print(f"  Saved: {out_filename}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted([f for f in os.listdir(args.json_dir) if f.endswith(".json")])

    if not json_files:
        print(f"[ERROR] No .json files found in {args.json_dir}")
        exit(1)

    print(f"Found {len(json_files)} JSON files. Processing...\n")
    missing_imgs = []

    for json_file in json_files:
        stem = os.path.splitext(json_file)[0]          # 'GL123_345'
        json_path = os.path.join(args.json_dir, json_file)
        img_path  = os.path.join(args.img_dir, stem + ".png")

        if not os.path.exists(img_path):
            print(f"  [WARNING] No matching image for {json_file} (expected: {img_path})")
            missing_imgs.append(json_file)
            continue

        print(f"Processing: {stem}")
        process_single(json_path, img_path, args.output_dir, args.alpha)

    # Audit summary
    print(f"\nDone. Processed {len(json_files) - len(missing_imgs)}/{len(json_files)} files.")
    if missing_imgs:
        print(f"[AUDIT] {len(missing_imgs)} JSON(s) had no matching image:")
        for f in missing_imgs:
            print(f"  - {f}")
