"""Generate 4-panel comparison PNGs using CellViT-SAM-H predictions (not CellViT-256).

Replaces the Phase 1 CellViT-256 qualitative figures with SAM-H equivalents.
Output: 3 PNGs in /local/data/.../qualitative_sam_h/ that will replace
the Phase 1 versions used in Figures 4.6, 4.7, 4.8.
"""
import sys
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage import measure
from PIL import Image

HD = Path("/local/data/sc23sp2/dissertation/heavy_data")
IMG_DIR = HD / "images_full_fold3"
GT_DIR = HD / "images_full_fold3" / "ground_truth_masks"
HN_DIR = HD / "outputs_full" / "hovernet_fold3"
SAM_DIR = HD / "outputs_full" / "cellvit_sam_h_fold3"
OUT_DIR = HD / "qualitative_sam_h"
OUT_DIR.mkdir(exist_ok=True)

# Class colours — match the Phase 1 scheme so the dissertation reads consistently
CLASS_COLORS = {
    1: '#E62828',  # Neoplastic - red
    2: '#1B68C2',  # Inflammatory - blue
    3: '#2A9134',  # Connective - green
    4: '#F0C419',  # Dead - yellow
    5: '#E67E22',  # Epithelial - orange
}
CLASS_NAMES = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'}


def draw_pred_contours(ax, img, inst_map, class_per_inst):
    ax.imshow(img)
    ax.axis('off')
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        mask = (inst_map == inst_id).astype(np.uint8)
        contours = measure.find_contours(mask, 0.5)
        cls = class_per_inst.get(int(inst_id), 1)
        color = CLASS_COLORS.get(cls, '#999999')
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.4)


def draw_gt_contours(ax, img, gt_insts, gt_cats):
    ax.imshow(img)
    ax.axis('off')
    for i in range(gt_insts.shape[0]):
        mask = gt_insts[i].astype(np.uint8)
        contours = measure.find_contours(mask, 0.5)
        cls = int(gt_cats[i]) + 1  # 0-indexed GT -> 1-indexed
        color = CLASS_COLORS.get(cls, '#999999')
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.4)


def make_panel(image_name):
    img = np.array(Image.open(IMG_DIR / f"{image_name}.png").convert('RGB'))
    gt_insts = np.load(GT_DIR / f"{image_name}_instances.npy")
    gt_cats = np.load(GT_DIR / f"{image_name}_categories.npy")
    hn = joblib.load(HN_DIR / f"{image_name}.dat")
    hn_map = np.asarray(hn['instance_map']).astype(np.int32)
    hn_cls = {int(iid): info.get('type', 0) for iid, info in hn['nuclei'].items()}
    sam = joblib.load(SAM_DIR / f"{image_name}.dat")
    sam_map = np.asarray(sam['instance_map']).astype(np.int32)
    sam_cls = {int(iid): info.get('type', 0) for iid, info in sam['nuclei'].items()}

    gt_n = gt_insts.shape[0]
    hn_n = hn['n_nuclei']
    sam_n = sam['n_nuclei']
    tissue = image_name.split('_')[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10.5))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Original: {image_name} ({tissue})", fontsize=11)
    axes[0, 0].axis('off')
    draw_gt_contours(axes[0, 1], img, gt_insts, gt_cats)
    axes[0, 1].set_title(f"Ground Truth ({gt_n} nuclei)", fontsize=11)
    draw_pred_contours(axes[1, 0], img, hn_map, hn_cls)
    axes[1, 0].set_title(f"HoVer-Net ({hn_n} nuclei)", fontsize=11)
    draw_pred_contours(axes[1, 1], img, sam_map, sam_cls)
    axes[1, 1].set_title(f"CellViT-SAM-H ({sam_n} nuclei)", fontsize=11)

    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor='black', label=CLASS_NAMES[c])
               for c in [1, 2, 3, 4, 5]]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    out_path = OUT_DIR / f"{image_name}_sam_h.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path.name}: GT={gt_n}, HN={hn_n}, SAM-H={sam_n}")
    return gt_n, hn_n, sam_n


print("Regenerating qualitative images using CellViT-SAM-H predictions...\n")
results = {}
for name in ['kidney_0000', 'lung_0000', 'breast_0000']:
    print(f"{name}:")
    results[name] = make_panel(name)

print("\n=== Numbers to put in your captions ===")
for name, (gt, hn, sam) in results.items():
    tissue = name.split('_')[0].capitalize()
    print(f"{tissue}: GT={gt}, HoVer-Net detects {hn}, CellViT-SAM-H detects {sam}")

print(f"\nOutput: {OUT_DIR}")
