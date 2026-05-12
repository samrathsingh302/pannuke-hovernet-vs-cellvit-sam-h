"""Official PanNuke metric pipeline.

1. Build (N, 256, 256, 5) masks for GT, HoVer-Net, CellViT-SAM-H
2. GT-vs-GT smoke test → must score 1.000
3. Real metrics for both models using PanNuke-metrics utils
4. Bootstrap 95% CIs (n=1000)
"""

import sys, os, json, time
from pathlib import Path
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

# Import the OFFICIAL PanNuke-metrics functions
sys.path.insert(0, str(Path("PanNuke-metrics").resolve()))
from utils import get_fast_pq, remap_label, binarize

# ──────────── Paths ────────────
HD = Path("/local/data/sc23sp2/dissertation/heavy_data")
GT_DIR = HD / "images_full_fold3" / "ground_truth_masks"
HN_DIR = HD / "outputs_full" / "hovernet_fold3"
SAM_DIR = HD / "outputs_full" / "cellvit_sam_h_fold3"
META_FILE = HD / "images_full_fold3" / "metadata.json"
OUT_FILE = HD / "metrics_results_official.json"

# ──────────── PanNuke-metrics canonical names (from run.py) ────────────
PANNUKE_TISSUES = [
    'Adrenal_gland','Bile-duct','Bladder','Breast','Cervix','Colon',
    'Esophagus','HeadNeck','Kidney','Liver','Lung','Ovarian',
    'Pancreatic','Prostate','Skin','Stomach','Testis','Thyroid','Uterus'
]

# Map HF dataset names → PanNuke canonical names
HF_TO_PANNUKE = {
    'Adrenal Gland':'Adrenal_gland','Bile Duct':'Bile-duct','Bladder':'Bladder',
    'Breast':'Breast','Cervix':'Cervix','Colon':'Colon','Esophagus':'Esophagus',
    'Head & Neck':'HeadNeck','Kidney':'Kidney','Liver':'Liver','Lung':'Lung',
    'Ovarian':'Ovarian','Pancreatic':'Pancreatic','Prostate':'Prostate','Skin':'Skin',
    'Stomach':'Stomach','Testis':'Testis','Thyroid':'Thyroid','Uterus':'Uterus',
}

CLASS_NAMES = ['Neoplastic','Inflammatory','Connective','Dead','NonNeoplasticEpithelial']

# ──────────── Mask builders ────────────

def build_gt_masks(name):
    """HF GT → (256, 256, 5) per-class instance maps."""
    instances = np.load(GT_DIR / f"{name}_instances.npy")  # (N, 256, 256)
    categories = np.load(GT_DIR / f"{name}_categories.npy")  # (N,) int 0-4
    masks = np.zeros((256, 256, 5), dtype=np.int16)
    for c in range(5):
        idx = np.where(categories == c)[0]
        for new_id, i in enumerate(idx, start=1):
            mask = instances[i].astype(bool)
            if mask.sum() > 0:
                masks[mask, c] = new_id
    return masks

def build_pred_masks(dat_path):
    """Prediction .dat → (256, 256, 5)."""
    data = joblib.load(dat_path)
    inst_map = np.asarray(data['instance_map']).astype(np.int32)
    nuclei = data.get('nuclei', {})
    by_class = {c: [] for c in range(5)}
    for inst_id_raw, info in nuclei.items():
        try: inst_id = int(inst_id_raw)
        except (TypeError, ValueError): continue
        cls = info.get('type', 0)
        if cls and 1 <= cls <= 5:
            by_class[cls - 1].append(inst_id)  # cls 1-5 → channel 0-4
    masks = np.zeros((256, 256, 5), dtype=np.int16)
    for channel, inst_ids in by_class.items():
        for new_id, inst_id in enumerate(inst_ids, start=1):
            mask = inst_map == inst_id
            if mask.sum() > 0:
                masks[mask, channel] = new_id
    return masks

# ──────────── Per-image PQ (called from pool) ────────────

# Globals set in main, inherited by fork workers
TRUE_MASKS = None
PRED_MASKS = None

def compute_pq_at(i):
    """Compute bPQ and per-class PQ for image i. Uses globals."""
    true_i = TRUE_MASKS[i].astype('int32')
    pred_i = PRED_MASKS[i].astype('int32')
    bpq = np.nan
    pq_class = np.full(5, np.nan)
    
    # Binary PQ
    true_bin = binarize(true_i[:, :, :5])
    pred_bin = binarize(pred_i[:, :, :5])
    if len(np.unique(true_bin)) > 1:
        [_, _, bpq], _ = get_fast_pq(true_bin, pred_bin)
    
    # Per-class PQ
    for j in range(5):
        true_tmp = remap_label(true_i[:, :, j])
        pred_tmp = remap_label(pred_i[:, :, j])
        if len(np.unique(true_tmp)) > 1:
            [_, _, pq], _ = get_fast_pq(true_tmp, pred_tmp)
            pq_class[j] = pq
    
    return i, bpq, pq_class

def compute_metrics_parallel(true, pred, n_workers=12):
    """Per-image metrics across all images (parallelised via fork)."""
    global TRUE_MASKS, PRED_MASKS
    TRUE_MASKS = true
    PRED_MASKS = pred
    
    N = true.shape[0]
    bpq_per_image = np.full(N, np.nan)
    pq_per_image_class = np.full((N, 5), np.nan)
    
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        n_processed = 0
        t0 = time.time()
        for i, bpq, pq_class in pool.imap_unordered(compute_pq_at, range(N), chunksize=10):
            bpq_per_image[i] = bpq
            pq_per_image_class[i] = pq_class
            n_processed += 1
            if n_processed % 500 == 0:
                el = time.time() - t0
                print(f"    [{n_processed}/{N}] {el:.0f}s, ETA {(N-n_processed)*el/n_processed/60:.1f}min", flush=True)
    
    return bpq_per_image, pq_per_image_class

# ──────────── Aggregation (matches run.py exactly) ────────────

def aggregate(bpq_per_image, pq_per_image_class, types):
    """PanNuke aggregation: mean over tissues of mean over images."""
    mpq_per_image = np.array([np.nanmean(row) for row in pq_per_image_class])
    
    tissue_bpq = {}
    tissue_mpq = {}
    for t in PANNUKE_TISSUES:
        idx = [i for i, x in enumerate(types) if x == t]
        if len(idx) == 0:
            tissue_bpq[t] = float('nan')
            tissue_mpq[t] = float('nan')
        else:
            tissue_bpq[t] = float(np.nanmean(bpq_per_image[idx]))
            tissue_mpq[t] = float(np.nanmean(mpq_per_image[idx]))
    
    overall_bpq = float(np.nanmean(list(tissue_bpq.values())))
    overall_mpq = float(np.nanmean(list(tissue_mpq.values())))
    
    class_pq = {CLASS_NAMES[c]: float(np.nanmean(pq_per_image_class[:, c]))
                for c in range(5)}
    
    return {
        'bPQ': overall_bpq, 'mPQ': overall_mpq,
        'tissue_bPQ': tissue_bpq, 'tissue_mPQ': tissue_mpq,
        'class_PQ': class_pq,
    }

def bootstrap_ci(bpq_per_image, pq_per_image_class, types, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(bpq_per_image)
    types_arr = np.array(types)
    
    bpq_samples, mpq_samples = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        agg = aggregate(bpq_per_image[idx], pq_per_image_class[idx], types_arr[idx])
        bpq_samples.append(agg['bPQ'])
        mpq_samples.append(agg['mPQ'])
    
    bpq = np.array(bpq_samples)
    mpq = np.array(mpq_samples)
    return {
        'bPQ_ci': [float(np.nanpercentile(bpq, 2.5)), float(np.nanpercentile(bpq, 97.5))],
        'bPQ_mean': float(np.nanmean(bpq)),
        'bPQ_std': float(np.nanstd(bpq)),
        'mPQ_ci': [float(np.nanpercentile(mpq, 2.5)), float(np.nanpercentile(mpq, 97.5))],
        'mPQ_mean': float(np.nanmean(mpq)),
        'mPQ_std': float(np.nanstd(mpq)),
    }

# ──────────── Main ────────────

print("=== Loading metadata ===", flush=True)
with open(META_FILE) as f:
    meta_data = json.load(f)
metadata = meta_data.get('metadata', meta_data) if isinstance(meta_data, dict) else meta_data

# Determine usable images (have GT + HN + SAM-H)
hn_files = {p.stem for p in HN_DIR.glob("*.dat")}
sam_files = {p.stem for p in SAM_DIR.glob("*.dat")}
usable = []
for item in metadata:
    name = item['filename'].replace('.png', '')
    if (name in hn_files and name in sam_files
        and (GT_DIR / f"{name}_instances.npy").exists()):
        usable.append(item)
N = len(usable)
print(f"  Usable images: {N} / {len(metadata)}")

# Tissue strings in PanNuke canonical format
hf_tissues = [item.get('tissue_name', 'unknown') for item in usable]
types = np.array([HF_TO_PANNUKE.get(t, t) for t in hf_tissues])
print(f"  Tissue distribution:")
for t in PANNUKE_TISSUES:
    n = int((types == t).sum())
    print(f"    {t:<20}: {n}")

# ──────────── Build mask arrays ────────────
print(f"\n=== Building (N, 256, 256, 5) masks for {N} images ===", flush=True)
t0 = time.time()
true_masks = np.zeros((N, 256, 256, 5), dtype=np.int16)
hn_masks = np.zeros((N, 256, 256, 5), dtype=np.int16)
sam_masks = np.zeros((N, 256, 256, 5), dtype=np.int16)

for i, item in enumerate(usable):
    name = item['filename'].replace('.png', '')
    true_masks[i] = build_gt_masks(name)
    hn_masks[i] = build_pred_masks(HN_DIR / f"{name}.dat")
    sam_masks[i] = build_pred_masks(SAM_DIR / f"{name}.dat")
    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{N}] {time.time()-t0:.0f}s", flush=True)

print(f"  Done in {(time.time()-t0)/60:.1f} min")
print(f"  Memory: gt={true_masks.nbytes/1e9:.2f} GB, hn={hn_masks.nbytes/1e9:.2f} GB, sam={sam_masks.nbytes/1e9:.2f} GB")

# ──────────── GT-vs-GT smoke test ────────────
print("\n" + "="*70)
print("=== GT-vs-GT SMOKE TEST (must score 1.000) ===")
print("="*70, flush=True)
print(f"Running on first 50 images for speed...")
t0 = time.time()
sub_bpq, sub_pq = compute_metrics_parallel(true_masks[:50], true_masks[:50], n_workers=12)
sub_agg = aggregate(sub_bpq, sub_pq, types[:50])
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  bPQ = {sub_agg['bPQ']:.6f}  (must be 1.000000)")
print(f"  mPQ = {sub_agg['mPQ']:.6f}  (must be 1.000000)")
if abs(sub_agg['bPQ'] - 1.0) > 1e-4 or abs(sub_agg['mPQ'] - 1.0) > 1e-4:
    print(f"  ❌ FAILED — adapter has a bug. NOT proceeding with real metrics.")
    sys.exit(1)
print(f"  ✅ PASSED — adapter format is correct, can trust real metrics")

# ──────────── HoVer-Net real metrics ────────────
print("\n" + "="*70)
print("=== HoVer-Net real metrics ===")
print("="*70, flush=True)
t0 = time.time()
hn_bpq, hn_pq_class = compute_metrics_parallel(true_masks, hn_masks, n_workers=12)
print(f"  Done in {(time.time()-t0)/60:.1f} min")
agg_hn = aggregate(hn_bpq, hn_pq_class, types)
print(f"\n  bPQ = {agg_hn['bPQ']:.4f}")
print(f"  mPQ = {agg_hn['mPQ']:.4f}")
print(f"  Per-class PQ:")
for c, v in agg_hn['class_PQ'].items():
    print(f"    {c:<26} = {v:.4f}")

# ──────────── SAM-H real metrics ────────────
print("\n" + "="*70)
print("=== CellViT-SAM-H real metrics ===")
print("="*70, flush=True)
t0 = time.time()
sam_bpq, sam_pq_class = compute_metrics_parallel(true_masks, sam_masks, n_workers=12)
print(f"  Done in {(time.time()-t0)/60:.1f} min")
agg_sam = aggregate(sam_bpq, sam_pq_class, types)
print(f"\n  bPQ = {agg_sam['bPQ']:.4f}")
print(f"  mPQ = {agg_sam['mPQ']:.4f}")
print(f"  Per-class PQ:")
for c, v in agg_sam['class_PQ'].items():
    print(f"    {c:<26} = {v:.4f}")

# ──────────── Bootstrap CIs ────────────
print("\n" + "="*70)
print("=== Bootstrap 95% CIs (n=1000, image-level resampling) ===")
print("="*70, flush=True)

t0 = time.time()
print("HoVer-Net...", flush=True)
hn_ci = bootstrap_ci(hn_bpq, hn_pq_class, types, n_bootstrap=1000)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  bPQ = {agg_hn['bPQ']:.4f}  95% CI [{hn_ci['bPQ_ci'][0]:.4f}, {hn_ci['bPQ_ci'][1]:.4f}]")
print(f"  mPQ = {agg_hn['mPQ']:.4f}  95% CI [{hn_ci['mPQ_ci'][0]:.4f}, {hn_ci['mPQ_ci'][1]:.4f}]")

t0 = time.time()
print("\nCellViT-SAM-H...", flush=True)
sam_ci = bootstrap_ci(sam_bpq, sam_pq_class, types, n_bootstrap=1000)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  bPQ = {agg_sam['bPQ']:.4f}  95% CI [{sam_ci['bPQ_ci'][0]:.4f}, {sam_ci['bPQ_ci'][1]:.4f}]")
print(f"  mPQ = {agg_sam['mPQ']:.4f}  95% CI [{sam_ci['mPQ_ci'][0]:.4f}, {sam_ci['mPQ_ci'][1]:.4f}]")

# ──────────── Per-tissue table ────────────
print("\n" + "="*70)
print("=== PER-TISSUE bPQ and mPQ ===")
print("="*70)
print(f"{'tissue':<18} {'n':>5} {'HN bPQ':>10} {'SAM bPQ':>10} {'Δ bPQ':>9} {'HN mPQ':>10} {'SAM mPQ':>10} {'Δ mPQ':>9}")
print("-"*92)
for t in PANNUKE_TISSUES:
    n = int((types == t).sum())
    hb = agg_hn['tissue_bPQ'].get(t, np.nan)
    hm = agg_hn['tissue_mPQ'].get(t, np.nan)
    sb = agg_sam['tissue_bPQ'].get(t, np.nan)
    sm = agg_sam['tissue_mPQ'].get(t, np.nan)
    db = sb - hb if not (np.isnan(sb) or np.isnan(hb)) else np.nan
    dm = sm - hm if not (np.isnan(sm) or np.isnan(hm)) else np.nan
    db_s = f"{db:>+9.4f}" if not np.isnan(db) else f"{'nan':>9}"
    dm_s = f"{dm:>+9.4f}" if not np.isnan(dm) else f"{'nan':>9}"
    print(f"{t:<18} {n:>5} {hb:>10.4f} {sb:>10.4f} {db_s} {hm:>10.4f} {sm:>10.4f} {dm_s}")

# ──────────── Save ────────────
results = {
    'gt_smoke_test': {'bPQ': sub_agg['bPQ'], 'mPQ': sub_agg['mPQ'], 'passed': True},
    'hovernet': {**agg_hn, 'bootstrap_ci': hn_ci},
    'cellvit_sam_h': {**agg_sam, 'bootstrap_ci': sam_ci},
    'n_images': N,
    'metric_implementation': 'official PanNuke-metrics (Gamper et al. 2020)',
    'bootstrap_n': 1000,
}

def json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [json_safe(x) for x in obj]
    return obj

with open(OUT_FILE, 'w') as f:
    json.dump(json_safe(results), f, indent=2)

print(f"\n  Saved to {OUT_FILE}")
print(f"\n=== ALL DONE ===")
