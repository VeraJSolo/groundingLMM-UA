import os
import json
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Save binary masks for all inference outputs.")
    parser.add_argument("--json_dir",   required=True, help="Folder containing inference .json files (e.g. GL123_345.json)")
    parser.add_argument("--output_dir", required=True, help="Folder to save binary mask images")
    return parser.parse_args()

def process_single(json_path, output_dir):
    stem = os.path.splitext(os.path.basename(json_path))[0]  # e.g. 'GL123_345'

    with open(json_path) as f:
        data = json.load(f)

    if not data.get("pred_masks"):
        print(f"  [SKIP] No masks in {json_path}")
        return

    for i, pred_mask in enumerate(data["pred_masks"]):
        rle = pred_mask
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")

        # Decode RLE to 2D binary array (H, W)
        mask_array = maskUtils.decode(rle)

        # Save as black/white image
        # e.g. GL123_345_mask0.png, GL123_345_mask1.png ...
        out_filename = f"{stem}_mask{i}.png"
        out_path = os.path.join(output_dir, out_filename)
        Image.fromarray((mask_array * 255).astype(np.uint8)).save(out_path)
        print(f"  Saved: {out_filename}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted([f for f in os.listdir(args.json_dir) if f.endswith(".json")])

    if not json_files:
        print(f"[ERROR] No .json files found in {args.json_dir}")
        exit(1)

    print(f"Found {len(json_files)} JSON files. Processing...\n")

    for json_file in json_files:
        stem = os.path.splitext(json_file)[0]
        json_path = os.path.join(args.json_dir, json_file)
        print(f"Processing: {stem}")
        process_single(json_path, args.output_dir)

    print("\nDone.")
