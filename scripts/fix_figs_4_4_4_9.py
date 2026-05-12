"""Fix Fig 4.4 and Fig 4.9 by using correct get_fast_pq unpacking."""
import sys, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("PanNuke-metrics").resolve()))
from utils import get_fast_pq

HD = Path("/local/data/sc23sp2/dissertation/heavy_data")
GT_DIR = HD / "images_full_fold3" / "ground_truth_masks"
HN_DIR = HD / "outputs_full" / "hovernet_fold3"
SAM_DIR = HD / "outputs_full" / "cellvit_sam_h_fold3"
META_FILE = HD / "images_full_fold3" / "metadata.json"
OUT_DIR = HD / "dissertation_figures"

# Load metadata
raw = json.load(open(META_FILE))
records = raw['metadata']
CLASS_NAMES = raw['class_names_official']
N_CLASSES = len(CLASS_NAMES)

meta = {}
for r in records:
    name = r['filename'].replace('.png', '')
    meta[name] = {'tissue_name': r['tissue_name'], 'n_nuclei': r['n_nuclei']}

tissue_to_imgs = defaultdict(list)
for name, info in meta.items():
    tissue_to_imgs[info['tissue_name']].append(name)
TISSUES = sorted(tissue_to_imgs.keys())


def load_gt(name):
    insts = np.load(GT_DIR / f"{name}_instances.npy")
    cats = np.load(GT_DIR / f"{name}_categories.npy")
    return insts, cats


def load_pred(pred_dir, name):
    d = joblib.load(Path(pred_dir) / f"{name}.dat")
    inst_map = np.asarray(d['instance_map']).astype(np.int32)
    cls = {int(iid): info.get('type', info.get('class', 0))
           for iid, info in d['nuclei'].items()}
    return inst_map, cls


# ============================================================
# Diagnose get_fast_pq return signature
# ============================================================
print("Testing get_fast_pq return signature...")
test_gt = np.zeros((256, 256), dtype=np.int32)
test_gt[10:30, 10:30] = 1
test_gt[40:60, 40:60] = 2
result = get_fast_pq(test_gt, test_gt)
print(f"  result type: {type(result).__name__}, len: {len(result)}")
print(f"  result[0] = {result[0]}")
print(f"  result[0][2] (PQ) = {result[0][2]}  (should be 1.0)")


# ============================================================
# FIGURE 4.4 — Per-tissue x per-class PQ heatmap (FIXED)
# ============================================================
print("\n[Fig 4.4] Recomputing per-tissue x per-class PQ (fixed unpacking)...")


def build_class_mask(insts, cats, c):
    m = np.zeros((256, 256), dtype=np.int32)
    nid = 1
    for i in range(insts.shape[0]):
        if cats[i] == c:
            m[insts[i]] = nid
            nid += 1
    return m


def build_pred_class_mask(inst_map, cls, c):
    m = np.zeros((256, 256), dtype=np.int32)
    nid = 1
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        if cls.get(int(iid), 0) == c + 1:
            m[inst_map == iid] = nid
            nid += 1
    return m


def per_img_per_class_fixed(args):
    name, pred_dir = args
    try:
        insts, cats = load_gt(name)
        inst_map, cls = load_pred(pred_dir, name)
        out = []
        for c in range(N_CLASSES):
            gt_m = build_class_mask(insts, cats, c)
            pr_m = build_pred_class_mask(inst_map, cls, c)
            if gt_m.max() == 0 and pr_m.max() == 0:
                out.append(np.nan)
                continue
            r = get_fast_pq(gt_m, pr_m)  # FIXED: just access by index
            out.append(r[0][2])
        return name, out
    except Exception as e:
        return name, [np.nan] * N_CLASSES


def grid_for(pred_dir, label):
    print(f"  computing {label}...")
    args_list = [(n, str(pred_dir)) for n in meta.keys()]
    per_img = {}
    with Pool(min(12, cpu_count())) as pool:
        for i, (name, vals) in enumerate(pool.imap_unordered(per_img_per_class_fixed, args_list, chunksize=20), 1):
            per_img[name] = vals
            if i % 500 == 0:
                print(f"    [{i}/{len(args_list)}]")
    grid = np.zeros((len(TISSUES), N_CLASSES))
    for ti, t in enumerate(TISSUES):
        for c in range(N_CLASSES):
            vals = [per_img[n][c] for n in tissue_to_imgs[t]
                    if not np.isnan(per_img[n][c])]
            grid[ti, c] = float(np.mean(vals)) if vals else 0.0
    return grid


hn_grid = grid_for(HN_DIR, "HoVer-Net")
sam_grid = grid_for(SAM_DIR, "CellViT-SAM-H")

print(f"\n  HN grid summary: min={hn_grid.min():.3f}, max={hn_grid.max():.3f}, mean={hn_grid.mean():.3f}")
print(f"  SAM grid summary: min={sam_grid.min():.3f}, max={sam_grid.max():.3f}, mean={sam_grid.mean():.3f}")

np.save(OUT_DIR / "per_tissue_class_pq_hn.npy", hn_grid)
np.save(OUT_DIR / "per_tissue_class_pq_sam.npy", sam_grid)

fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
for ax, grid, ttl in zip(axes, [hn_grid, sam_grid], ["HoVer-Net", "CellViT-SAM-H"]):
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=0.85, aspect='auto')
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticks(range(len(TISSUES)))
    ax.set_yticklabels(TISSUES, fontsize=9)
    ax.set_title(ttl, fontsize=11)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=axes, label='PQ', shrink=0.8)
plt.suptitle('Per-tissue x per-class PQ', fontsize=13, y=1.0)
plt.savefig(OUT_DIR / "fig_4_4_per_tissue_class_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  saved fig 4.4")


# ============================================================
# FIGURE 4.9 — Per-image bPQ scatter (FIXED)
# ============================================================
print("\n[Fig 4.9] Computing per-image bPQ scatter (fixed)...")


def per_img_bpq_fixed(args):
    name, pred_dir = args
    try:
        insts, _ = load_gt(name)
        inst_map, _ = load_pred(pred_dir, name)
        gt_bin = np.zeros((256, 256), dtype=np.int32)
        nid = 1
        for i in range(insts.shape[0]):
            gt_bin[insts[i]] = nid
            nid += 1
        if gt_bin.max() == 0 and inst_map.max() == 0:
            return name, np.nan
        r = get_fast_pq(gt_bin, inst_map.astype(np.int32))  # FIXED
        return name, r[0][2]
    except Exception as e:
        return name, np.nan


print("  computing HN...")
with Pool(min(12, cpu_count())) as pool:
    hn_b = dict(pool.imap_unordered(per_img_bpq_fixed,
                                     [(n, str(HN_DIR)) for n in meta.keys()],
                                     chunksize=20))
print("  computing SAM...")
with Pool(min(12, cpu_count())) as pool:
    sam_b = dict(pool.imap_unordered(per_img_bpq_fixed,
                                      [(n, str(SAM_DIR)) for n in meta.keys()],
                                      chunksize=20))

joblib.dump({"hn": hn_b, "sam": sam_b}, OUT_DIR / "per_image_bpq.joblib")

# Diagnostics
hn_valid = [v for v in hn_b.values() if not np.isnan(v)]
sam_valid = [v for v in sam_b.values() if not np.isnan(v)]
print(f"\n  HN valid: {len(hn_valid)}/{len(hn_b)}, mean={np.mean(hn_valid):.4f}")
print(f"  SAM valid: {len(sam_valid)}/{len(sam_b)}, mean={np.mean(sam_valid):.4f}")

xs, ys, tlabels = [], [], []
for name in meta.keys():
    h, s = hn_b.get(name, np.nan), sam_b.get(name, np.nan)
    if not np.isnan(h) and not np.isnan(s):
        xs.append(h); ys.append(s); tlabels.append(meta[name]['tissue_name'])

xs, ys = np.array(xs), np.array(ys)
total = len(xs)
print(f"  paired (both valid): {total}")

if total == 0:
    print("  ERROR: No valid pairs - cannot generate scatter")
else:
    unique_t = sorted(set(tlabels))
    cmap = plt.cm.get_cmap('tab20', len(unique_t))
    t2c = {t: cmap(i) for i, t in enumerate(unique_t)}
    colors = [t2c[t] for t in tlabels]
    above = int((ys > xs).sum())
    corr = float(np.corrcoef(xs, ys)[0, 1])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(xs, ys, c=colors, s=12, alpha=0.5, edgecolor='none')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Equal performance (y = x)')
    ax.text(0.05, 0.95,
            f'CellViT-SAM-H higher: {above}/{total} ({100*above/total:.1f}%)\n'
            f'Pearson r = {corr:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'))
    ax.set_xlabel('HoVer-Net per-image bPQ', fontsize=11)
    ax.set_ylabel('CellViT-SAM-H per-image bPQ', fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')
    handles = [plt.scatter([], [], c=[t2c[t]], label=t, s=20) for t in unique_t]
    fig.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc='center left',
               fontsize=8, frameon=True, title="Tissue")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_4_9_per_image_bpq_scatter.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  saved fig 4.9")
    print(f"  KEY NUMBERS: above-diag {above}/{total} ({100*above/total:.1f}%), Pearson r = {corr:.3f}")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
for f in sorted(OUT_DIR.glob("*.png")):
    sz = f.stat().st_size // 1024
    print(f"  {f.name}  ({sz} KB)")
