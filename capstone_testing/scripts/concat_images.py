"""
concat_images.py — Side-by-side image concatenation pre-processor for GLaMM inference

Takes a dataset/reference folder where each image group has its own subfolder.
All .png files inside a subfolder are concatenated horizontally into a single
output image named after the subfolder.

The output files use the subfolder name as their filename (GL092_228.png),
which matches the naming convention expected by infer.py, decode_mask.py,
and maskimage.py downstream.

Usage:
    python concat_images.py \
        --input_dir  /path/to/dataset/reference \
        --output_dir /path/to/concatenated_images \
        --separator  10         # optional white gap (pixels) between panels
        --resize_height 1024    # optional: resize all panels to same height before concat

Folder structure expected:
    input_dir/                       (e.g. dataset/reference/)
        GL092_228/
            GL092_228_001_001.png
            GL092_228_001_002.png
            GL092_228_001_003.png    <- variable number of images per folder
        GL093_001/
            GL093_001_001_001.png
            GL093_001_001_002.png
        ...

Output produced:
    output_dir/
        GL092_228.png    <- all images in GL092_228/ concatenated side by side
        GL093_001.png    <- all images in GL093_001/ concatenated side by side
"""

import os
import argparse
import numpy as np
from PIL import Image
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate component images side-by-side for GLaMM inference."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root reference folder containing one subfolder per image group "
                             "(e.g. dataset/reference/). Each subfolder name becomes the output filename.")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to save concatenated output images.")
    parser.add_argument("--separator", type=int, default=0,
                        help="Width in pixels of white gap between panels (default: 0 = no gap).")
    parser.add_argument("--resize_height", type=int, default=None,
                        help="If set, resize all panels to this height before concatenating "
                             "(aspect ratio preserved). Recommended if component images have "
                             "different sizes. Default: auto-match to most common height.")
    parser.add_argument("--ext", type=str, default=".png",
                        help="File extension to look for inside subfolders (default: .png).")
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Group discovery - one subfolder = one group
# ------------------------------------------------------------------------------

def discover_groups(input_dir, ext):
    """
    Scan input_dir for subfolders. For each subfolder, collect all image files
    sorted alphabetically (so _001_001 < _001_002, etc.).

    Returns:
        list of (group_name, [full_path, ...]) sorted by group_name
    """
    groups = []

    for entry in sorted(os.listdir(input_dir)):
        entry_path = os.path.join(input_dir, entry)

        if not os.path.isdir(entry_path):
            continue  # skip loose files at the top level

        images = sorted([
            f for f in os.listdir(entry_path)
            if f.lower().endswith(ext.lower())
        ])

        if not images:
            print(f"  [WARNING] Subfolder '{entry}' contains no {ext} files - skipping.")
            continue

        full_paths = [os.path.join(entry_path, f) for f in images]
        groups.append((entry, full_paths))

    return groups


# ------------------------------------------------------------------------------
# Image concatenation helpers
# ------------------------------------------------------------------------------

def resize_to_height(img_array, target_height):
    """Resize (H, W, C) array to target_height, preserving aspect ratio."""
    h, w = img_array.shape[:2]
    if h == target_height:
        return img_array
    scale   = target_height / h
    new_w   = max(1, int(w * scale))
    resized = Image.fromarray(img_array).resize((new_w, target_height), Image.LANCZOS)
    return np.array(resized)


def concatenate_side_by_side(image_paths, separator_px=0, resize_height=None):
    """
    Load images from paths, optionally resize to a common height, and
    concatenate horizontally with an optional white separator.

    Returns a PIL Image.
    """
    arrays = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB"))
        if resize_height is not None:
            img = resize_to_height(img, resize_height)
        arrays.append(img)

    # Ensure all panels have the same height
    heights = [a.shape[0] for a in arrays]
    if len(set(heights)) > 1:
        target_h = Counter(heights).most_common(1)[0][0]
        print(f"    [WARNING] Mixed heights {set(heights)} - "
              f"auto-resizing all to most common height={target_h}px. "
              f"Use --resize_height to override.")
        arrays = [resize_to_height(a, target_h) for a in arrays]

    if separator_px > 0:
        h   = arrays[0].shape[0]
        sep = np.ones((h, separator_px, 3), dtype=np.uint8) * 255  # white gap
        panels = []
        for i, arr in enumerate(arrays):
            panels.append(arr)
            if i < len(arrays) - 1:
                panels.append(sep)
        combined = np.concatenate(panels, axis=1)
    else:
        combined = np.concatenate(arrays, axis=1)

    return Image.fromarray(combined)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning subfolders in: {args.input_dir}\n")
    groups = discover_groups(args.input_dir, args.ext)

    if not groups:
        print(f"[ERROR] No subfolders with {args.ext} files found in {args.input_dir}")
        exit(1)

    panel_counts = [len(paths) for _, paths in groups]
    print(f"Found {len(groups)} image groups")
    print(f"Panels per group: min={min(panel_counts)}, "
          f"max={max(panel_counts)}, avg={sum(panel_counts)/len(panel_counts):.1f}")
    if args.resize_height:
        print(f"Resize height   : {args.resize_height}px")
    if args.separator:
        print(f"Separator       : {args.separator}px white gap between panels")
    print()

    failed = []

    for group_name, full_paths in groups:
        output_path = os.path.join(args.output_dir, group_name + args.ext)
        n = len(full_paths)

        try:
            if n == 1:
                # Only one image in folder - pass through unchanged
                Image.open(full_paths[0]).convert("RGB").save(output_path)
                print(f"  [PASS-THROUGH] {group_name}{args.ext}  (1 image in folder)")
                continue

            print(f"  {group_name}  ({n} panels)")
            for p in full_paths:
                print(f"    + {os.path.basename(p)}")

            result = concatenate_side_by_side(
                full_paths,
                separator_px=args.separator,
                resize_height=args.resize_height,
            )
            result.save(output_path)
            w, h = result.size
            print(f"    -> {group_name}{args.ext}  ({w}x{h}px)\n")

        except Exception as e:
            print(f"  [ERROR] '{group_name}': {e}")
            failed.append(group_name)

    # --------------------------------------------------------------------------
    print("=" * 52)
    print(f"Done. {len(groups) - len(failed)} / {len(groups)} groups processed.")
    if failed:
        print(f"[AUDIT] {len(failed)} group(s) failed:")
        for s in failed:
            print(f"  - {s}")
    print(f"\nConcatenated images saved to: {args.output_dir}")
    print()
    print("Next step - run inference on the concatenated images:")
    print(f"  python eval/gcg/infer.py \\")
    print(f"      --hf_model_path  <your_model> \\")
    print(f"      --img_dir        {args.output_dir} \\")
    print(f"      --prompt_dir     <your_prompt_dir> \\")
    print(f"      --output_dir     <your_output_dir>")
    print("=" * 52)
