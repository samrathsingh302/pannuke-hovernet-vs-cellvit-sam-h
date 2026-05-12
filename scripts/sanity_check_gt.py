"""Compare HoVer-Net detections to ground truth for breast_00.

Handles both possible formats of the categories array.
"""

import joblib
import numpy as np
from collections import Counter

# Load HoVer-Net predictions
pred = joblib.load("./outputs/smoke_test/0.dat")
pred_classes = Counter(d['type'] for d in pred.values())

# Load ground truth
gt_inst = np.load("./images/ground_truth_masks/breast_00_instances.npy")
gt_cat = np.load("./images/ground_truth_masks/breast_00_categories.npy")

print(f"gt_inst shape: {gt_inst.shape}, dtype: {gt_inst.dtype}")
print(f"  unique values: {len(np.unique(gt_inst))} (incl. background)")
print(f"gt_cat  shape: {gt_cat.shape}, dtype: {gt_cat.dtype}")
print(f"  unique values: {np.unique(gt_cat)}")
print()

# Count GT nuclei
unique_ids = np.unique(gt_inst)
unique_ids = unique_ids[unique_ids > 0]  # exclude background
gt_n_nuclei = len(unique_ids)

# Determine GT class per nucleus
gt_class_per_nucleus = []
if gt_cat.ndim == 1:
    # Per-instance class vector format
    print("Format detected: gt_cat is 1D (per-instance classes)")
    for cls in gt_cat:
        if cls > 0:
            gt_class_per_nucleus.append(int(cls))
elif gt_cat.ndim == 2:
    # 2D class mask format - look up class for each instance
    print("Format detected: gt_cat is 2D (class mask)")
    for inst_id in unique_ids:
        mask = (gt_inst == inst_id)
        classes_in_mask = gt_cat[mask]
        classes_in_mask = classes_in_mask[classes_in_mask > 0]
        if len(classes_in_mask) > 0:
            most_common = int(np.bincount(classes_in_mask).argmax())
            gt_class_per_nucleus.append(most_common)
elif gt_cat.ndim == 3:
    # PanNuke original: (H, W, 6) one-hot per class
    print("Format detected: gt_cat is 3D (one-hot per class)")
    for inst_id in unique_ids:
        mask = (gt_inst == inst_id)
        # Sum over the spatial extent
        counts = gt_cat[..., :].sum(axis=(0, 1)) if gt_cat.shape[-1] == 6 else None
        # Determine class within mask
        for c in range(gt_cat.shape[-1]):
            if gt_cat[..., c][mask].any():
                gt_class_per_nucleus.append(c)
                break

gt_class_dist = Counter(gt_class_per_nucleus)

# PanNuke class names
class_names = {0: 'Background', 1: 'Neoplastic', 2: 'Inflammatory',
               3: 'Connective', 4: 'Dead', 5: 'Epithelial'}

print()
print("="*60)
print(f"COMPARISON: HoVer-Net vs Ground Truth on breast_00.png")
print("="*60)
print(f"\nTotal nuclei:")
print(f"  Predicted:    {len(pred):3d}")
print(f"  Ground truth: {gt_n_nuclei:3d}")
print(f"\nClass distribution:")
print(f"  {'Class':<20} {'Predicted':<12} {'Ground Truth':<12}")
for cid in range(1, 6):
    name = class_names.get(cid, f'Class{cid}')
    p = pred_classes.get(cid, 0)
    g = gt_class_dist.get(cid, 0)
    print(f"  {name:<20} {p:<12} {g:<12}")

ratio = len(pred) / gt_n_nuclei if gt_n_nuclei > 0 else 0
print(f"\nDetection rate: {ratio*100:.1f}% (Predicted/GT)")
