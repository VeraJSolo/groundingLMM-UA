import os
import json
import argparse
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against ground truth binary masks.")
    parser.add_argument("--json_dir",  required=True,
                        help="Folder of inference .json files")
    parser.add_argument("--gt_dir",    required=True,
                        help="Folder of ground truth binary mask .png files"
                             "White (255) = foreground, Black (0) = background.")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to save results CSV and summary.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold to count a prediction as a 'hit' (default: 0.5)")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------------

def compute_iou(pred_binary, gt_binary):
    """
    Compute IoU between two boolean/binary numpy arrays.
    Returns 0.0 if both masks are empty (avoids divide-by-zero).
    """
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union        = np.logical_or(pred_binary,  gt_binary).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)
    
    
def compute_precision(pred_binary, gt_binary):
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    target_area        = gt_binary.sum()
    if target_area == 0:
        return 0.0
    return float(intersection) / float(target_area)
    
    
def compute_recall(pred_binary, gt_binary):
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    predicted_area        = pred_binary.sum()
    if predicted_area == 0:
        return 0.0
    return float(intersection) / float(predicted_area)
    
    
def compute_f1(pred_binary, gt_binary):
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    target_area        = gt_binary.sum()
    if target_area == 0:
        p = 0.0
    else:
        p = float(intersection) / float(target_area)
    predicted_area        = pred_binary.sum()
    if predicted_area == 0:
        r = 0.0
    else:
        r = float(intersection) / float(predicted_area)
    numer = 2 * p * r
    denom = p + r
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def compute_dice(pred_binary, gt_binary):
    """
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Ranges 0–1. More sensitive than IoU for small masks.
    Returns 0.0 if both masks are empty.
    """
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    denom = pred_binary.sum() + gt_binary.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection) / float(denom)


# -----------------------------------------------------------------------------
# Per-image evaluation
# -----------------------------------------------------------------------------

def evaluate_single(json_path, gt_path):
    """
    Compare all predicted masks in a JSON against a single ground truth mask.
    Returns the best-matching prediction's IoU, precision, recall, F1, and Dice, 
    plus how many masks the model produced.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Load ground truth — white pixels (>127) = foreground
    gt_array = np.array(Image.open(gt_path).convert("L"))
    gt_binary = gt_array > 127

    pred_masks = data.get("pred_masks", [])
    if not pred_masks:
        return {
            "n_pred_masks": 0,
            "best_iou":     0.0,
            "best_dice":    0.0,
            "best_p":     0.0,
            "best_r":    0.0,
            "best_f1":     0.0,
            "best_mask_idx": None,
            "caption":      data.get("caption", ""),
            "phrases":      data.get("phrases", []),
            "note":         "no predicted masks",
        }

    best_iou   = 0.0
    best_dice  = 0.0
    best_p = 0.0
    best_r = 0.0
    best_f1 = 0.0
    best_idx   = 0

    for i, pred_mask in enumerate(pred_masks):
        rle = pred_mask
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")

        pred_array  = maskUtils.decode(rle)   # shape: (H, W), values 0/1
        pred_binary = pred_array.astype(bool)

        iou  = compute_iou(pred_binary,  gt_binary)
        dice = compute_dice(pred_binary, gt_binary)
        precision = compute_precision(pred_binary, gt_binary)
        recall = compute_recall(pred_binary, gt_binary)
        f1 = compute_f1(pred_binary, gt_binary)

        if iou > best_iou:
            best_iou  = iou
            best_dice = dice
            best_idx  = i
            best_p = precision
            best_r = recall
            best_f1 = f1

    return {
        "n_pred_masks":  len(pred_masks),
        "best_iou":      round(best_iou,  4),
        "best_dice":     round(best_dice, 4),
        "best_p":    round(best_p, 4),
        "best_r":   round(best_r, 4),
        "best_f1":  round(best_f1, 4),
        "best_mask_idx": best_idx,
        "caption":       data.get("caption", ""),
        "phrases":       data.get("phrases", []),
        "note":          "",
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted([f for f in os.listdir(args.json_dir) if f.endswith(".json")])
    if not json_files:
        print(f"[ERROR] No .json files found in {args.json_dir}")
        exit(1)

    print(f"Found {len(json_files)} JSON files. Evaluating...\n")

    rows          = []   # for CSV
    missing_gt    = []
    all_ious      = []
    all_dices     = []
    all_p = []
    all_r = []
    all_f1 = []
    hits          = 0    # predictions with IoU >= threshold

    for json_file in json_files:
        stem      = os.path.splitext(json_file)[0]          # 'GL123_345'
        json_path = os.path.join(args.json_dir, json_file)
        gt_path   = os.path.join(args.gt_dir,   stem + ".png")

        if not os.path.exists(gt_path):
            print(f"  [WARNING] No GT mask for {json_file} (expected: {gt_path})")
            missing_gt.append(stem)
            continue

        result = evaluate_single(json_path, gt_path)

        all_ious.append(result["best_iou"])
        all_dices.append(result["best_dice"])
        all_p.append(result["best_p"])
        all_r.append(result["best_r"])
        all_f1.append(result["best_f1"])
        if result["best_iou"] >= args.iou_threshold:
            hits += 1

        rows.append({
            "image_id":      stem,
            "best_iou":      result["best_iou"],
            "best_dice":     result["best_dice"],
            "best_p":     result["best_p"],
            "best_r":     result["best_r"],
            "best_f1":     result["best_f1"],
            "best_mask_idx": result["best_mask_idx"],
            "n_pred_masks":  result["n_pred_masks"],
            "caption":       result["caption"],
            "phrases":       "; ".join(result["phrases"]),
            "note":          result["note"],
        })

        status = f"IoU={result['best_iou']:.3f}  Dice={result['best_dice']:.3f} P={result['best_p']:.3f}  R={result['best_r']:.3f} F1={result['best_f1']:.3f}"
        if result["note"]:
            status += f"  [{result['note']}]"
        print(f"  {stem:30s}  {status}")

    # -------------------------------------------------------------------------
    # Aggregate summary
    # -------------------------------------------------------------------------
    n_evaluated = len(all_ious)
    mean_iou    = round(float(np.mean(all_ious)),  4) if all_ious  else 0.0
    mean_dice   = round(float(np.mean(all_dices)), 4) if all_dices else 0.0
    mean_p    = round(float(np.mean(all_p)),  4) if all_ious  else 0.0
    mean_r   = round(float(np.mean(all_r)), 4) if all_dices else 0.0
    mean_f1    = round(float(np.mean(all_f1)),  4) if all_ious  else 0.0
    hit_rate    = round(hits / n_evaluated, 4)         if n_evaluated else 0.0

    summary_lines = [
        "=" * 52,
        "EVALUATION SUMMARY",
        "=" * 52,
        f"Images evaluated   : {n_evaluated}",
        f"Missing GT masks   : {len(missing_gt)}",
        f"Mean IoU           : {mean_iou}",
        f"Mean Dice          : {mean_dice}",
        f"Mean P          : {mean_p}",
        f"Mean R          : {mean_r}",
        f"Mean F1           : {mean_f1}",
        f"Hit rate (IoU>={args.iou_threshold:.2f}): {hit_rate}  ({hits}/{n_evaluated})",
        "=" * 52,
        "",
        "Metric guide:",
        "  IoU  >= 0.75  → strong segmentation",
        "  IoU  >= 0.50  → acceptable (COCO standard threshold)",
        "  IoU  <  0.25  → poor overlap, review these images",
        "  Dice >= 0.80  → strong, especially for small masks",
    ]

    print("\n" + "\n".join(summary_lines))

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    # Per-image CSV
    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    fieldnames = ["image_id", "best_iou", "best_dice", "best_p", "best_r", "best_f1", "best_mask_idx",
                  "n_pred_masks", "caption", "phrases", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-image results saved to : {csv_path}")

    # Summary text file
    summary_path = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
        if missing_gt:
            f.write("\nMissing GT masks:\n")
            for s in missing_gt:
                f.write(f"  - {s}\n")
    print(f"Summary saved to           : {summary_path}")