"""Generate 4-panel comparison visualisations: Original | GT | HoVer-Net | CellViT.

Saves PNG per image to outputs/comparison/<tissue>_<idx>.png
"""

import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
img_dir = Path("./images")
gt_dir = Path("./images/ground_truth_masks")
hn_dir = Path("./outputs/hovernet_all")
cv_dir = Path("./outputs/cellvit_all")
out_dir = Path("./outputs/comparison")
out_dir.mkdir(parents=True, exist_ok=True)

# PanNuke class names + colours
# HoVer-Net + CellViT use 1-indexed: 1=Neop, 2=Inflam, 3=Conn, 4=Dead, 5=Epi
# PanNuke GT (RationAI HF) is 0-indexed: 0=Neop, 1=Inflam, 2=Conn, 3=Dead, 4=Epi
PRED_CLASS_NAMES = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'}
PRED_COLORS = {1: 'red', 2: 'blue', 3: 'lime', 4: 'yellow', 5: 'orange'}

# GT 0-indexed → 1-indexed mapping for unified colouring
GT_TO_PRED = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

def draw_pred_overlay(ax, img_arr, pred_dict, contour_key='contour'):
    """Draw image + colored contours from a HoVer-Net or CellViT prediction dict."""
    ax.imshow(img_arr)
    n_per_class = {c: 0 for c in PRED_CLASS_NAMES}
    for nucleus_id, info in pred_dict.items():
        contour = info[contour_key]
        cls = info.get('type', 0)
        if cls in PRED_COLORS:
            n_per_class[cls] = n_per_class.get(cls, 0) + 1
            # contour shape: (N, 2) - [(x, y), ...]
            ax.plot(contour[:, 0], contour[:, 1],
                    color=PRED_COLORS[cls], linewidth=1.0)
    # Total
    total = sum(n_per_class.values())
    return total, n_per_class

def draw_gt_overlay(ax, img_arr, gt_inst, gt_cat):
    """Draw GT contours on image. gt_inst shape: (N, H, W) bool, gt_cat shape: (N,)."""
    ax.imshow(img_arr)
    n_per_class = {c: 0 for c in PRED_CLASS_NAMES}
    n = gt_inst.shape[0]
    
    from skimage import measure
    for i in range(n):
        mask = gt_inst[i]
        if not mask.any():
            continue
        cls_0idx = int(gt_cat[i])
        cls_pred = GT_TO_PRED.get(cls_0idx, 0)
        if cls_pred not in PRED_COLORS:
            continue
        # Find contour of the mask
        contours = measure.find_contours(mask.astype(np.uint8), 0.5)
        for c in contours:
            n_per_class[cls_pred] = n_per_class.get(cls_pred, 0) + 1
            # find_contours returns (row, col) -> (y, x); need to plot as (x, y)
            ax.plot(c[:, 1], c[:, 0], color=PRED_COLORS[cls_pred], linewidth=1.0)
            break  # only outer contour
    total = sum(n_per_class.values())
    return total, n_per_class

# Map HoVer-Net image index → tissue_idx name (via metadata)
import json
with open(img_dir / "metadata.json") as f:
    metadata = json.load(f)

# HoVer-Net outputs are stored as outputs/hovernet_all/<idx>/0.dat
# CellViT outputs are stored as outputs/cellvit_all/<tissue>_<idx>.dat
# We need to match them via metadata

print(f"Generating comparisons for {len(metadata)} images...")
all_stats = []

for hn_idx, meta in enumerate(metadata):
    img_name = Path(meta['filename']).stem  # e.g. "breast_00"
    tissue = meta['tissue_name']
    
    # Load original image
    img_path = img_dir / meta['filename']
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    
    # Load HoVer-Net prediction
    hn_dat = hn_dir / f"{hn_idx}.dat"
    hn_pred = joblib.load(hn_dat) if hn_dat.exists() else {}
    
    # Load CellViT prediction
    cv_dat = cv_dir / f"{img_name}.dat"
    cv_data = joblib.load(cv_dat)
    cv_pred = cv_data['nuclei']
    
    # Load ground truth
    gt_inst = np.load(gt_dir / f"{img_name}_instances.npy")
    gt_cat = np.load(gt_dir / f"{img_name}_categories.npy")
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0,0].imshow(img_arr)
    axes[0,0].set_title(f"Original: {img_name} ({tissue})")
    axes[0,0].axis('off')
    
    gt_total, gt_classes = draw_gt_overlay(axes[0,1], img_arr, gt_inst, gt_cat)
    axes[0,1].set_title(f"Ground Truth ({gt_total} nuclei)")
    axes[0,1].axis('off')
    
    hn_total, hn_classes = draw_pred_overlay(axes[1,0], img_arr, hn_pred, 'contour')
    axes[1,0].set_title(f"HoVer-Net ({hn_total} nuclei)")
    axes[1,0].axis('off')
    
    cv_total, cv_classes = draw_pred_overlay(axes[1,1], img_arr, cv_pred, 'contour')
    axes[1,1].set_title(f"CellViT ({cv_total} nuclei)")
    axes[1,1].axis('off')
    
    # Legend
    legend_handles = [mpatches.Patch(color=PRED_COLORS[c], label=PRED_CLASS_NAMES[c])
                      for c in [1, 2, 3, 4, 5]]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_dir / f"{img_name}.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # Track stats
    all_stats.append({
        'image': img_name,
        'tissue': tissue,
        'gt': gt_total, 'hn': hn_total, 'cv': cv_total,
        'gt_classes': gt_classes,
        'hn_classes': hn_classes,
        'cv_classes': cv_classes,
    })
    
    if (hn_idx + 1) % 10 == 0:
        print(f"  {hn_idx + 1}/{len(metadata)} done...")

# Save stats
joblib.dump(all_stats, out_dir / "all_stats.dat")
print(f"\n=== Saved {len(all_stats)} comparison images to {out_dir} ===")
print(f"=== Saved stats to {out_dir / 'all_stats.dat'} ===")

# Summary
print("\n=== Aggregate totals ===")
print(f"{'Source':<15} {'Total':<10} {'Avg/img':<10}")
for src, key in [('Ground Truth', 'gt'), ('HoVer-Net', 'hn'), ('CellViT', 'cv')]:
    total = sum(s[key] for s in all_stats)
    avg = total / len(all_stats)
    print(f"{src:<15} {total:<10} {avg:<10.1f}")
