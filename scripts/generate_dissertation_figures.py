"""Generate all remaining dissertation figures from the inference outputs."""
import os, sys, json
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
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path("PanNuke-metrics").resolve()))
from utils import get_fast_pq

HD = Path("/local/data/sc23sp2/dissertation/heavy_data")
GT_DIR = HD / "images_full_fold3" / "ground_truth_masks"
HN_DIR = HD / "outputs_full" / "hovernet_fold3"
SAM_DIR = HD / "outputs_full" / "cellvit_sam_h_fold3"
META_FILE = HD / "images_full_fold3" / "metadata.json"
OUT_DIR = HD / "dissertation_figures"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Load metadata (new structure: {'metadata': [list of records], ...})
# ============================================================
print("Loading metadata...")
raw = json.load(open(META_FILE))
records = raw['metadata']
CLASS_NAMES = raw['class_names_official']  # 5 names
N_CLASSES = len(CLASS_NAMES)

# image basename -> {tissue_name, n_nuclei}
meta = {}
for r in records:
    name = r['filename'].replace('.png', '')
    meta[name] = {
        'tissue_name': r['tissue_name'],
        'n_nuclei': r['n_nuclei'],
    }

tissue_to_imgs = defaultdict(list)
for name, info in meta.items():
    tissue_to_imgs[info['tissue_name']].append(name)
TISSUES = sorted(tissue_to_imgs.keys())
print(f"  {len(meta)} images, {len(TISSUES)} tissues, {N_CLASSES} classes")
print(f"  tissues: {TISSUES}")
print(f"  classes: {CLASS_NAMES}")


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
# FIGURE 2.1 — Class frequency by tissue
# ============================================================
print("\n[Fig 2.1] Class frequency heatmap...")


def count_one(name):
    cats = np.load(GT_DIR / f"{name}_categories.npy")
    return name, [int((cats == c).sum()) for c in range(N_CLASSES)]


with Pool(min(12, cpu_count())) as pool:
    counts = dict(pool.imap_unordered(count_one, list(meta.keys()), chunksize=50))

freq = np.zeros((len(TISSUES), N_CLASSES), dtype=int)
for ti, tissue in enumerate(TISSUES):
    for img in tissue_to_imgs[tissue]:
        for c in range(N_CLASSES):
            freq[ti, c] += counts[img][c]

fig, ax = plt.subplots(figsize=(8.5, 7))
norm = mcolors.LogNorm(vmin=max(1, freq[freq > 0].min()), vmax=freq.max())
im = ax.imshow(freq, aspect='auto', cmap='YlOrRd', norm=norm)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_yticks(range(len(TISSUES)))
ax.set_yticklabels(TISSUES)
ax.set_xlabel('Class', fontsize=11)
ax.set_ylabel('Tissue', fontsize=11)
for i in range(len(TISSUES)):
    for j in range(N_CLASSES):
        v = freq[i, j]
        c = 'white' if v > freq.max() * 0.4 else 'black'
        ax.text(j, i, str(v), ha='center', va='center', fontsize=8, color=c)
plt.colorbar(im, ax=ax, label='Nuclei count (log scale)')
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_2_1_class_freq_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  saved   total nuclei: {int(freq.sum())}")


# ============================================================
# FIGURE 2.2 — Nuclei per patch boxplot
# ============================================================
print("\n[Fig 2.2] Nuclei per patch boxplot...")
per_patch = {t: [] for t in TISSUES}
for name, info in meta.items():
    per_patch[info['tissue_name']].append(info['n_nuclei'])

medians = {t: float(np.median(per_patch[t])) for t in TISSUES}
sorted_t = sorted(TISSUES, key=lambda t: medians[t])
data_sorted = [per_patch[t] for t in sorted_t]

fig, ax = plt.subplots(figsize=(11, 5.5))
bp = ax.boxplot(data_sorted, labels=sorted_t, patch_artist=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='.', markerfacecolor='gray', markersize=4, alpha=0.5))
for patch in bp['boxes']:
    patch.set_facecolor('#88A8C8')
    patch.set_alpha(0.7)
ax.set_xticklabels(sorted_t, rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Tissue (sorted by median)', fontsize=11)
ax.set_ylabel('Nuclei per patch', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_2_2_nuclei_per_patch.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"  saved")


# ============================================================
# FIGURES 4.2 / 4.3 — Confusion matrices (IoU >= 0.5 matching)
# ============================================================
print("\n[Figs 4.2/4.3] Confusion matrices...")


def match_one(args):
    name, pred_dir = args
    pairs = []
    try:
        gt_insts, gt_cats = load_gt(name)
        inst_map, cls = load_pred(pred_dir, name)
        if gt_insts.shape[0] == 0:
            return pairs
        pred_ids = np.unique(inst_map)
        pred_ids = pred_ids[pred_ids > 0]
        if len(pred_ids) == 0:
            return pairs
        pred_masks = np.stack([(inst_map == pid) for pid in pred_ids])
        gt_areas = gt_insts.sum(axis=(1, 2))
        pred_areas = pred_masks.sum(axis=(1, 2))
        for gi in range(gt_insts.shape[0]):
            inter = (pred_masks & gt_insts[gi]).sum(axis=(1, 2))
            union = gt_areas[gi] + pred_areas - inter
            iou = inter / np.maximum(union, 1)
            best = int(iou.argmax())
            if iou[best] >= 0.5:
                gt_c = int(gt_cats[gi])
                pred_c = int(cls.get(int(pred_ids[best]), 0))
                if 1 <= pred_c <= 5:
                    pairs.append((gt_c, pred_c - 1))
    except Exception as e:
        pass
    return pairs


def build_cm(pred_dir, label):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    args_list = [(n, str(pred_dir)) for n in meta.keys()]
    print(f"  computing {label}...")
    with Pool(min(12, cpu_count())) as pool:
        for i, pairs in enumerate(pool.imap_unordered(match_one, args_list, chunksize=20), 1):
            for g, p in pairs:
                cm[g, p] += 1
            if i % 500 == 0:
                print(f"    [{i}/{len(args_list)}]")
    return cm


cm_hn = build_cm(HN_DIR, "HoVer-Net")
cm_sam = build_cm(SAM_DIR, "CellViT-SAM-H")
np.save(OUT_DIR / "confmat_hovernet.npy", cm_hn)
np.save(OUT_DIR / "confmat_cellvit_sam_h.npy", cm_sam)


def plot_cm(cm, title, path):
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted class', fontsize=11)
    ax.set_ylabel('Ground-truth class', fontsize=11)
    ax.set_title(title, fontsize=12)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = cm_norm[i, j]
            c = 'white' if v > 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}\n({cm[i,j]})',
                    ha='center', va='center', fontsize=9, color=c)
    plt.colorbar(im, ax=ax, label='Row-normalised proportion')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


plot_cm(cm_hn, "HoVer-Net confusion matrix (matched at IoU>=0.5)",
        OUT_DIR / "fig_4_2_confmat_hovernet.png")
plot_cm(cm_sam, "CellViT-SAM-H confusion matrix (matched at IoU>=0.5)",
        OUT_DIR / "fig_4_3_confmat_cellvit_sam_h.png")
print(f"  HN matched: {cm_hn.sum()}; SAM matched: {cm_sam.sum()}")


# ============================================================
# FIGURE 4.4 — Per-tissue x per-class PQ heatmap
# ============================================================
print("\n[Fig 4.4] Per-tissue x per-class PQ heatmap...")


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


def per_img_per_class(args):
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
            pq, _, _ = get_fast_pq(gt_m, pr_m)
            out.append(pq[2])
        return name, out
    except Exception:
        return name, [np.nan] * N_CLASSES


def grid_for(pred_dir, label):
    print(f"  computing {label}...")
    args_list = [(n, str(pred_dir)) for n in meta.keys()]
    per_img = {}
    with Pool(min(12, cpu_count())) as pool:
        for i, (name, vals) in enumerate(pool.imap_unordered(per_img_per_class, args_list, chunksize=20), 1):
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
print(f"  saved")


# ============================================================
# FIGURE 4.9 — Per-image bPQ scatter
# ============================================================
print("\n[Fig 4.9] Per-image bPQ scatter...")


def per_img_bpq(args):
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
        pq, _, _ = get_fast_pq(gt_bin, inst_map.astype(np.int32))
        return name, pq[2]
    except Exception:
        return name, np.nan


print("  computing HN...")
with Pool(min(12, cpu_count())) as pool:
    hn_b = dict(pool.imap_unordered(per_img_bpq,
                                     [(n, str(HN_DIR)) for n in meta.keys()],
                                     chunksize=20))
print("  computing SAM...")
with Pool(min(12, cpu_count())) as pool:
    sam_b = dict(pool.imap_unordered(per_img_bpq,
                                      [(n, str(SAM_DIR)) for n in meta.keys()],
                                      chunksize=20))

joblib.dump({"hn": hn_b, "sam": sam_b}, OUT_DIR / "per_image_bpq.joblib")

xs, ys, tlabels = [], [], []
for name in meta.keys():
    h, s = hn_b.get(name, np.nan), sam_b.get(name, np.nan)
    if not np.isnan(h) and not np.isnan(s):
        xs.append(h); ys.append(s); tlabels.append(meta[name]['tissue_name'])

xs, ys = np.array(xs), np.array(ys)
unique_t = sorted(set(tlabels))
cmap = plt.cm.get_cmap('tab20', len(unique_t))
t2c = {t: cmap(i) for i, t in enumerate(unique_t)}
colors = [t2c[t] for t in tlabels]

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(xs, ys, c=colors, s=12, alpha=0.5, edgecolor='none')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Equal performance (y = x)')
above = int((ys > xs).sum())
total = len(xs)
corr = float(np.corrcoef(xs, ys)[0, 1])
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
print(f"  saved   above-diag {above}/{total} ({100*above/total:.1f}%)   r={corr:.3f}")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("  ALL FIGURES GENERATED")
print("=" * 60)
for f in sorted(OUT_DIR.glob("*.png")):
    sz = f.stat().st_size // 1024
    print(f"  {f.name}  ({sz} KB)")
print("=" * 60)
