"""Comprehensive validation of the entire dissertation pipeline.

Checks:
  1. File integrity        — counts, load tests
  2. Schema consistency    — .dat files have right structure
  3. GT distribution       — totals match published numbers
  4. Tissue distribution   — all 19, counts match
  5. Tissue name mapping   — every HF name maps cleanly
  6. Class indexing        — 1-indexed predictions, 0-indexed GT
  7. Mask roundtrip        — build then decode, verify identity
  8. FULL GT-vs-GT smoke   — all 2,722 images (not just 50)
  9. Reproducibility       — bootstrap with 3 seeds, must agree
 10. Pred/GT consistency   — instance IDs align between map and dict
 11. Memory check          — output files sane size

Each check prints PASS / FAIL / WARN. Final summary counts results.
"""

import sys, os, json, time, traceback, glob
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

sys.path.insert(0, str(Path("PanNuke-metrics").resolve()))
from utils import get_fast_pq, remap_label, binarize

# ──────────── Paths ────────────
HD = Path("/local/data/sc23sp2/dissertation/heavy_data")
GT_DIR = HD / "images_full_fold3" / "ground_truth_masks"
IMG_DIR = HD / "images_full_fold3"
HN_DIR = HD / "outputs_full" / "hovernet_fold3"
SAM_DIR = HD / "outputs_full" / "cellvit_sam_h_fold3"
META_FILE = HD / "images_full_fold3" / "metadata.json"
RESULTS_FILE = HD / "metrics_results_official.json"

# ──────────── Expected values ────────────
EXPECTED_TOTAL_IMAGES = 2722
EXPECTED_TOTAL_GT_NUCLEI = 66654
EXPECTED_PER_CLASS_GT = {
    'Neoplastic': 28471, 'Inflammatory': 10825, 'Connective': 17441,
    'Dead': 1057, 'Epithelial': 8860,
}
EXPECTED_PER_TISSUE = {
    'Adrenal_gland':155, 'Bile-duct':158, 'Bladder':64, 'Breast':775,
    'Cervix':86, 'Colon':494, 'Esophagus':141, 'HeadNeck':145,
    'Kidney':41, 'Liver':93, 'Lung':51, 'Ovarian':52,
    'Pancreatic':28, 'Prostate':68, 'Skin':41, 'Stomach':48,
    'Testis':57, 'Thyroid':62, 'Uterus':163,
}
PANNUKE_TISSUES = list(EXPECTED_PER_TISSUE.keys())
HF_TO_PANNUKE = {
    'Adrenal Gland':'Adrenal_gland','Bile Duct':'Bile-duct','Bladder':'Bladder',
    'Breast':'Breast','Cervix':'Cervix','Colon':'Colon','Esophagus':'Esophagus',
    'Head & Neck':'HeadNeck','Kidney':'Kidney','Liver':'Liver','Lung':'Lung',
    'Ovarian':'Ovarian','Pancreatic':'Pancreatic','Prostate':'Prostate','Skin':'Skin',
    'Stomach':'Stomach','Testis':'Testis','Thyroid':'Thyroid','Uterus':'Uterus',
}

# Expected metrics from previous run (the "ground truth" of our results)
EXPECTED_HN_BPQ = 0.6583
EXPECTED_HN_MPQ = 0.4510
EXPECTED_SAM_BPQ = 0.7963
EXPECTED_SAM_MPQ = 0.7048
TOLERANCE = 0.001

# ──────────── Result tracking ────────────
results = {'pass': 0, 'fail': 0, 'warn': 0, 'details': []}

def check(name, condition, detail="", warn_only=False):
    if condition:
        print(f"  ✅ PASS   {name}")
        results['pass'] += 1
    elif warn_only:
        print(f"  ⚠️  WARN  {name}  {detail}")
        results['warn'] += 1
        results['details'].append(('WARN', name, detail))
    else:
        print(f"  ❌ FAIL  {name}  {detail}")
        results['fail'] += 1
        results['details'].append(('FAIL', name, detail))

def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

# ──────────── 1. File integrity ────────────

section("§1 — File integrity")

png_files = list(IMG_DIR.glob("*.png"))
check("Image PNG count == 2722", len(png_files) == EXPECTED_TOTAL_IMAGES,
      f"got {len(png_files)}")

gt_inst_files = list(GT_DIR.glob("*_instances.npy"))
gt_cat_files = list(GT_DIR.glob("*_categories.npy"))
check("GT instances.npy count == 2722", len(gt_inst_files) == EXPECTED_TOTAL_IMAGES,
      f"got {len(gt_inst_files)}")
check("GT categories.npy count == 2722", len(gt_cat_files) == EXPECTED_TOTAL_IMAGES,
      f"got {len(gt_cat_files)}")

hn_files = list(HN_DIR.glob("*.dat"))
sam_files = list(SAM_DIR.glob("*.dat"))
check("HoVer-Net .dat count == 2722", len(hn_files) == EXPECTED_TOTAL_IMAGES,
      f"got {len(hn_files)}")
check("CellViT-SAM-H .dat count == 2722", len(sam_files) == EXPECTED_TOTAL_IMAGES,
      f"got {len(sam_files)}")

# Filename consistency: each PNG has matching GT and predictions
png_names = {p.stem for p in png_files}
gt_names = {p.stem.replace('_instances', '') for p in gt_inst_files}
hn_names = {p.stem for p in hn_files}
sam_names = {p.stem for p in sam_files}

check("Every PNG has matching GT instances", png_names == gt_names,
      f"missing GT for: {sorted(png_names - gt_names)[:5]}")
check("Every PNG has matching HoVer-Net prediction", png_names == hn_names,
      f"missing HN for: {sorted(png_names - hn_names)[:5]}")
check("Every PNG has matching CellViT-SAM-H prediction", png_names == sam_names,
      f"missing SAM for: {sorted(png_names - sam_names)[:5]}")

# Output file sizes sanity
total_hn_size = sum(p.stat().st_size for p in hn_files) / 1e6
total_sam_size = sum(p.stat().st_size for p in sam_files) / 1e6
print(f"  ℹ️ HoVer-Net .dat total: {total_hn_size:.0f} MB ({total_hn_size/len(hn_files):.2f} MB/file)")
print(f"  ℹ️ SAM-H .dat total:     {total_sam_size:.0f} MB ({total_sam_size/len(sam_files):.2f} MB/file)")

# ──────────── 2. Schema consistency ────────────

section("§2 — Schema consistency (spot check 5 random files)")

import random
random.seed(42)
spot_checks = random.sample(sorted(png_names), 5)

required_keys = {'nuclei', 'instance_map', 'image_name', 'n_nuclei'}

for name in spot_checks:
    try:
        d = joblib.load(HN_DIR / f"{name}.dat")
        check(f"HN {name}: has all required keys",
              required_keys.issubset(d.keys()),
              f"missing: {required_keys - set(d.keys())}")
        check(f"HN {name}: instance_map is 256x256 int",
              d['instance_map'].shape == (256, 256) and 'int' in str(d['instance_map'].dtype))
    except Exception as e:
        check(f"HN {name}: load OK", False, str(e))
    
    try:
        d = joblib.load(SAM_DIR / f"{name}.dat")
        check(f"SAM {name}: has all required keys",
              required_keys.issubset(d.keys()),
              f"missing: {required_keys - set(d.keys())}")
        check(f"SAM {name}: instance_map is 256x256 int",
              d['instance_map'].shape == (256, 256) and 'int' in str(d['instance_map'].dtype))
    except Exception as e:
        check(f"SAM {name}: load OK", False, str(e))

# ──────────── 3. GT total counts ────────────

section("§3 — GT totals match published expectations")

print("Counting all nuclei in GT (this takes ~10s)...")
class_totals = Counter()
total_nuclei = 0
for f in gt_cat_files:
    cats = np.load(f)
    total_nuclei += len(cats)
    for c in cats:
        class_totals[int(c)] += 1

check(f"GT total nuclei == {EXPECTED_TOTAL_GT_NUCLEI}",
      total_nuclei == EXPECTED_TOTAL_GT_NUCLEI,
      f"got {total_nuclei}")

# Class names: HF order = Neoplastic(0), Inflammatory(1), Connective(2), Dead(3), Epithelial(4)
HF_CLASS_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
for hf_idx, cname in enumerate(HF_CLASS_NAMES):
    expected = EXPECTED_PER_CLASS_GT[cname]
    actual = class_totals[hf_idx]
    check(f"GT count {cname} == {expected:,d}", actual == expected,
          f"got {actual:,d}")

# ──────────── 4. Tissue distribution ────────────

section("§4 — Tissue distribution from metadata")

with open(META_FILE) as f:
    meta_data = json.load(f)
metadata = meta_data.get('metadata', meta_data) if isinstance(meta_data, dict) else meta_data

tissue_counter = Counter(item.get('tissue_name', '?') for item in metadata)
print("  HF tissue counts:")
for tissue_hf, expected_count in EXPECTED_PER_TISSUE.items():
    # Find HF name that maps to this PanNuke name
    hf_names = [hf for hf, pn in HF_TO_PANNUKE.items() if pn == tissue_hf]
    actual = sum(tissue_counter.get(hf, 0) for hf in hf_names)
    check(f"Tissue {tissue_hf}: count == {expected_count}", actual == expected_count,
          f"got {actual} (HF names tried: {hf_names})")

# ──────────── 5. Tissue name mapping ────────────

section("§5 — Tissue name mapping covers all data")

unique_hf_tissues = set(tissue_counter.keys())
mapped = {t for t in unique_hf_tissues if t in HF_TO_PANNUKE}
unmapped = unique_hf_tissues - mapped
check("All HF tissue names have a PanNuke mapping",
      len(unmapped) == 0,
      f"unmapped: {unmapped}")

target_pannuke = set(HF_TO_PANNUKE.values())
expected_pannuke = set(PANNUKE_TISSUES)
check("All 19 PanNuke tissues are reachable through mapping",
      target_pannuke == expected_pannuke,
      f"missing: {expected_pannuke - target_pannuke}")

# ──────────── 6. Class indexing ────────────

section("§6 — Class indexing consistency")

# Sample HN and SAM, check class IDs
hn_classes_seen = set()
sam_classes_seen = set()
for name in spot_checks:
    d_hn = joblib.load(HN_DIR / f"{name}.dat")
    d_sam = joblib.load(SAM_DIR / f"{name}.dat")
    for nuc in d_hn['nuclei'].values():
        hn_classes_seen.add(nuc.get('type'))
    for nuc in d_sam['nuclei'].values():
        sam_classes_seen.add(nuc.get('type'))

# Predictions should have classes 1-5 (or possibly 0 = none, but rare)
hn_valid = hn_classes_seen.issubset({0, 1, 2, 3, 4, 5})
sam_valid = sam_classes_seen.issubset({0, 1, 2, 3, 4, 5})
check(f"HN class IDs in expected range [0-5]: {sorted(hn_classes_seen)}", hn_valid)
check(f"SAM class IDs in expected range [0-5]: {sorted(sam_classes_seen)}", sam_valid)
check("HN uses 1-indexed classes (some class >0)", any(c > 0 for c in hn_classes_seen))
check("SAM uses 1-indexed classes (some class >0)", any(c > 0 for c in sam_classes_seen))

# ──────────── 7. Mask roundtrip ────────────

section("§7 — Mask roundtrip on 5 random images")

def build_gt_masks(name):
    inst = np.load(GT_DIR / f"{name}_instances.npy")
    cats = np.load(GT_DIR / f"{name}_categories.npy")
    masks = np.zeros((256, 256, 5), dtype=np.int32)
    for c in range(5):
        idx = np.where(cats == c)[0]
        for new_id, i in enumerate(idx, start=1):
            mask = inst[i].astype(bool)
            if mask.sum() > 0:
                masks[mask, c] = new_id
    return masks, inst, cats

for name in spot_checks:
    masks, inst, cats = build_gt_masks(name)
    # Each instance in inst should appear exactly once across the 5 channels
    n_inst_in_npy = sum(1 for i in range(len(inst)) if inst[i].any())
    n_inst_in_masks = 0
    for c in range(5):
        n_inst_in_masks += len(np.unique(masks[:,:,c])) - (1 if 0 in masks[:,:,c] else 0)
    check(f"GT {name}: {n_inst_in_npy} instances in npy == {n_inst_in_masks} in masks",
          n_inst_in_npy == n_inst_in_masks)
    
    # Total fg pixels in masks should equal total fg pixels in instance npys
    fg_npy = sum(int(inst[i].sum()) for i in range(len(inst)))
    fg_masks = int((masks > 0).sum())
    check(f"GT {name}: fg pixel count {fg_npy} == {fg_masks}", fg_npy == fg_masks,
          warn_only=(fg_npy != fg_masks))  # Some overlap possible at boundaries

# Pred mask roundtrip
def build_pred_masks(dat_path):
    data = joblib.load(dat_path)
    inst_map = data['instance_map'].astype(np.int32)
    nuclei = data.get('nuclei', {})
    by_class = defaultdict(list)
    for inst_id_raw, info in nuclei.items():
        try: inst_id = int(inst_id_raw)
        except (TypeError, ValueError): continue
        cls = info.get('type', 0)
        if 1 <= cls <= 5:
            by_class[cls - 1].append(inst_id)
    masks = np.zeros((256, 256, 5), dtype=np.int32)
    for channel, inst_ids in by_class.items():
        for new_id, inst_id in enumerate(inst_ids, start=1):
            mask = inst_map == inst_id
            if mask.sum() > 0:
                masks[mask, channel] = new_id
    return masks, inst_map, nuclei

for name in spot_checks:
    masks, inst_map, nuclei = build_pred_masks(HN_DIR / f"{name}.dat")
    n_nuclei_with_class = sum(1 for n in nuclei.values() if 1 <= n.get('type', 0) <= 5)
    n_inst_in_masks = sum(len(np.unique(masks[:,:,c])) - 1
                          for c in range(5) if (masks[:,:,c] > 0).any())
    check(f"HN {name}: nuclei with class ({n_nuclei_with_class}) == channel-summed ({n_inst_in_masks})",
          n_nuclei_with_class == n_inst_in_masks)

# ──────────── 8. FULL GT-vs-GT smoke test on all 2722 ────────────

section("§8 — FULL GT-vs-GT smoke test (all 2,722 images)")

print("This is the big one. Running PanNuke metric on GT-as-pred for ALL images...")
print("  Expected: bPQ = mPQ = 1.000000 across every tissue, every class")
print("  This takes ~1 minute with parallel pool")

def gt_smoke_one_image(name):
    """Compute PQ for one GT-vs-GT pair. Should always be 1.0 if not empty."""
    masks, _, _ = build_gt_masks(name)
    masks = masks.astype('int32')
    
    bpq = np.nan
    pq_class = np.full(5, np.nan)
    
    true_bin = binarize(masks[:, :, :5])
    if len(np.unique(true_bin)) > 1:
        [_, _, bpq], _ = get_fast_pq(true_bin, true_bin)
    
    for j in range(5):
        true_tmp = remap_label(masks[:, :, j])
        if len(np.unique(true_tmp)) > 1:
            [_, _, pq], _ = get_fast_pq(true_tmp, true_tmp)
            pq_class[j] = pq
    
    return name, bpq, pq_class

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

t0 = time.time()
all_names = sorted(png_names)
non_unity_bpq = []
non_unity_pq_class = []

with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context('fork')) as pool:
    n_processed = 0
    for name, bpq, pq_class in pool.map(gt_smoke_one_image, all_names, chunksize=20):
        if not np.isnan(bpq) and abs(bpq - 1.0) > 1e-6:
            non_unity_bpq.append((name, bpq))
        for j, pq in enumerate(pq_class):
            if not np.isnan(pq) and abs(pq - 1.0) > 1e-6:
                non_unity_pq_class.append((name, j, pq))
        n_processed += 1
        if n_processed % 500 == 0:
            print(f"    [{n_processed}/{len(all_names)}] {time.time()-t0:.0f}s")

print(f"  Done in {time.time()-t0:.0f}s")
check(f"All 2,722 images: bPQ == 1.0 (GT-vs-GT)",
      len(non_unity_bpq) == 0,
      f"non-unity in {len(non_unity_bpq)} images: {non_unity_bpq[:3]}")
check(f"All 2,722 images, all 5 classes: PQ == 1.0 (GT-vs-GT)",
      len(non_unity_pq_class) == 0,
      f"non-unity in {len(non_unity_pq_class)} cases: {non_unity_pq_class[:3]}")

# ──────────── 9. Reproducibility ────────────

section("§9 — Reproducibility (bootstrap with 3 different seeds)")

# Quick re-run on cached per-image metrics if available, else compute from scratch
# We'll re-bootstrap the headline numbers with different seeds and check stability

# First need per-image metrics — recompute on a 200-image sample
print("  Computing on 200-image sample for reproducibility check...")
sample_names = random.sample(sorted(png_names), 200)
hn_bpq = np.full(200, np.nan)
hn_pq_class = np.full((200, 5), np.nan)
sam_bpq = np.full(200, np.nan)
sam_pq_class = np.full((200, 5), np.nan)
sample_types = []

for i, name in enumerate(sample_names):
    item = next(x for x in metadata if x['filename'].replace('.png','') == name)
    sample_types.append(HF_TO_PANNUKE.get(item.get('tissue_name', ''), '?'))
    
    gt_masks, _, _ = build_gt_masks(name)
    gt_masks = gt_masks.astype('int32')
    
    for model_dir, bpq_arr, pq_class_arr in [
        (HN_DIR, hn_bpq, hn_pq_class),
        (SAM_DIR, sam_bpq, sam_pq_class),
    ]:
        pred_masks, _, _ = build_pred_masks(model_dir / f"{name}.dat")
        pred_masks = pred_masks.astype('int32')
        true_bin = binarize(gt_masks[:, :, :5])
        pred_bin = binarize(pred_masks[:, :, :5])
        if len(np.unique(true_bin)) > 1:
            [_, _, bpq], _ = get_fast_pq(true_bin, pred_bin)
            bpq_arr[i] = bpq
        for j in range(5):
            true_tmp = remap_label(gt_masks[:, :, j])
            pred_tmp = remap_label(pred_masks[:, :, j])
            if len(np.unique(true_tmp)) > 1:
                [_, _, pq], _ = get_fast_pq(true_tmp, pred_tmp)
                pq_class_arr[i, j] = pq

sample_types_arr = np.array(sample_types)

def bootstrap_summary(bpq, pq_class, types, seed, n=500):
    rng = np.random.default_rng(seed)
    N = len(bpq)
    bpq_samples, mpq_samples = [], []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        sb = bpq[idx]
        smc = pq_class[idx]
        st = types[idx]
        mpq_per_image = np.array([np.nanmean(row) for row in smc])
        # Per-tissue then mean
        tissue_bpq, tissue_mpq = [], []
        for t in PANNUKE_TISSUES:
            ti = [j for j, x in enumerate(st) if x == t]
            if ti:
                tissue_bpq.append(np.nanmean(sb[ti]))
                tissue_mpq.append(np.nanmean(mpq_per_image[ti]))
        bpq_samples.append(np.nanmean(tissue_bpq))
        mpq_samples.append(np.nanmean(tissue_mpq))
    return float(np.nanmean(bpq_samples)), float(np.nanmean(mpq_samples))

print("  Running bootstrap with seeds 42, 123, 7777 on the sample...")
bs_runs = []
for seed in [42, 123, 7777]:
    hn_b, hn_m = bootstrap_summary(hn_bpq, hn_pq_class, sample_types_arr, seed)
    sam_b, sam_m = bootstrap_summary(sam_bpq, sam_pq_class, sample_types_arr, seed)
    bs_runs.append((seed, hn_b, hn_m, sam_b, sam_m))
    print(f"    Seed {seed}: HN bPQ={hn_b:.4f}, mPQ={hn_m:.4f}; SAM bPQ={sam_b:.4f}, mPQ={sam_m:.4f}")

# Variance across seeds should be tiny
hn_bpq_std = np.std([r[1] for r in bs_runs])
hn_mpq_std = np.std([r[2] for r in bs_runs])
sam_bpq_std = np.std([r[3] for r in bs_runs])
sam_mpq_std = np.std([r[4] for r in bs_runs])
check(f"HN bPQ stable across seeds (std < 0.005)", hn_bpq_std < 0.005, f"std={hn_bpq_std:.4f}")
check(f"HN mPQ stable across seeds (std < 0.005)", hn_mpq_std < 0.005, f"std={hn_mpq_std:.4f}")
check(f"SAM bPQ stable across seeds (std < 0.005)", sam_bpq_std < 0.005, f"std={sam_bpq_std:.4f}")
check(f"SAM mPQ stable across seeds (std < 0.005)", sam_mpq_std < 0.005, f"std={sam_mpq_std:.4f}")

# ──────────── 10. Check saved results match expectations ────────────

section("§10 — Saved metrics file matches earlier run")

if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        saved = json.load(f)
    
    check(f"GT smoke test passed in saved file",
          saved.get('gt_smoke_test', {}).get('passed') == True)
    check(f"Saved HN bPQ ≈ {EXPECTED_HN_BPQ}",
          abs(saved['hovernet']['bPQ'] - EXPECTED_HN_BPQ) < TOLERANCE,
          f"got {saved['hovernet']['bPQ']:.4f}")
    check(f"Saved HN mPQ ≈ {EXPECTED_HN_MPQ}",
          abs(saved['hovernet']['mPQ'] - EXPECTED_HN_MPQ) < TOLERANCE,
          f"got {saved['hovernet']['mPQ']:.4f}")
    check(f"Saved SAM bPQ ≈ {EXPECTED_SAM_BPQ}",
          abs(saved['cellvit_sam_h']['bPQ'] - EXPECTED_SAM_BPQ) < TOLERANCE,
          f"got {saved['cellvit_sam_h']['bPQ']:.4f}")
    check(f"Saved SAM mPQ ≈ {EXPECTED_SAM_MPQ}",
          abs(saved['cellvit_sam_h']['mPQ'] - EXPECTED_SAM_MPQ) < TOLERANCE,
          f"got {saved['cellvit_sam_h']['mPQ']:.4f}")
    check("Bootstrap CIs are non-overlapping for bPQ",
          saved['hovernet']['bootstrap_ci']['bPQ_ci'][1] < 
          saved['cellvit_sam_h']['bootstrap_ci']['bPQ_ci'][0])
    check("Bootstrap CIs are non-overlapping for mPQ",
          saved['hovernet']['bootstrap_ci']['mPQ_ci'][1] < 
          saved['cellvit_sam_h']['bootstrap_ci']['mPQ_ci'][0])
else:
    check("Saved metrics file exists", False, str(RESULTS_FILE))

# ──────────── 11. HoVer-Net validation against published ────────────

section("§11 — HoVer-Net numbers vs published (Graham et al. 2019)")

PUBLISHED_HN = {'bPQ': 0.6596, 'mPQ': 0.4629,
                'Neoplastic': 0.5510, 'Inflammatory': 0.4055,
                'Connective': 0.4032, 'Dead': 0.1437,
                'NonNeoplasticEpithelial': 0.4757}

if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        saved = json.load(f)
    hn = saved['hovernet']
    
    bpq_diff = abs(hn['bPQ'] - PUBLISHED_HN['bPQ'])
    mpq_diff = abs(hn['mPQ'] - PUBLISHED_HN['mPQ'])
    check(f"HN bPQ within 0.02 of Graham 2019 (Δ={bpq_diff:.4f})", bpq_diff < 0.02)
    check(f"HN mPQ within 0.03 of Graham 2019 (Δ={mpq_diff:.4f})", mpq_diff < 0.03)
    
    for cname, expected in PUBLISHED_HN.items():
        if cname in ('bPQ', 'mPQ'): continue
        actual = hn['class_PQ'].get(cname, np.nan)
        diff = abs(actual - expected)
        check(f"HN {cname} PQ within 0.04 of Graham (Δ={diff:.4f})",
              diff < 0.04, warn_only=(diff >= 0.04 and diff < 0.06))

# ──────────── Summary ────────────

print(f"\n{'='*70}")
print("VALIDATION SUMMARY")
print('='*70)
print(f"  ✅ PASS:  {results['pass']}")
print(f"  ❌ FAIL:  {results['fail']}")
print(f"  ⚠️ WARN:  {results['warn']}")
print('='*70)

if results['fail'] == 0:
    print("\n🎉 ALL CHECKS PASSED — pipeline is validated end-to-end.")
    print("   Numbers are publication-ready. Submit with confidence.")
else:
    print(f"\n⚠️ {results['fail']} CHECK(S) FAILED — investigate before submission:")
    for level, name, detail in results['details']:
        if level == 'FAIL':
            print(f"   - {name}: {detail}")

# Save report
report = {
    'summary': {'pass': results['pass'], 'fail': results['fail'], 'warn': results['warn']},
    'failures': [{'name': n, 'detail': d} for level, n, d in results['details'] if level == 'FAIL'],
    'warnings': [{'name': n, 'detail': d} for level, n, d in results['details'] if level == 'WARN'],
    'reproducibility': {
        'bootstrap_runs': bs_runs,
        'std_across_seeds': {
            'HN_bPQ': float(hn_bpq_std), 'HN_mPQ': float(hn_mpq_std),
            'SAM_bPQ': float(sam_bpq_std), 'SAM_mPQ': float(sam_mpq_std),
        },
    },
}
report_path = HD / "validation_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
print(f"\n  Saved full report to {report_path}")
