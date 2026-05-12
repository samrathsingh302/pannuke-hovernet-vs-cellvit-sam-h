"""Download full PanNuke fold 3 (2,722 images) to /local."""

import os
os.environ.setdefault("HF_HOME", "/local/data/sc23sp2/dissertation/hf_cache")

from datasets import load_dataset
from pathlib import Path
import numpy as np
from PIL import Image
import json
import time
import re

print("=== Loading PanNuke fold 3 ===")
t0 = time.time()
ds = load_dataset("RationAI/PanNuke", "default", split="fold3")
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Total fold 3 images: {len(ds)}")

# Get tissue and class names from the dataset's ClassLabel features
tissue_class_label = ds.features['tissue']
tissue_names_official = tissue_class_label.names
class_names_official = ds.features['categories'].feature.names
print(f"  Tissues: {tissue_names_official}")
print(f"  Classes: {class_names_official}")

OUT_BASE = Path("/local/data/sc23sp2/dissertation/heavy_data")
out_img_dir = OUT_BASE / "images_full_fold3"
out_gt_dir  = out_img_dir / "ground_truth_masks"
out_img_dir.mkdir(parents=True, exist_ok=True)
out_gt_dir.mkdir(exist_ok=True)

def sanitize(name):
    """Convert 'Head & Neck' -> 'head_and_neck'."""
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

# Skip if already saved
existing = list(out_img_dir.glob("*.png"))
if len(existing) >= len(ds):
    print(f"\n  Already saved {len(existing)} images, skipping export")
else:
    print(f"\n=== Exporting {len(ds)} images ===")
    t0 = time.time()
    metadata = []
    tissue_counts = {}
    
    for i, sample in enumerate(ds):
        tissue_id = int(sample["tissue"])
        tissue_name_safe = sanitize(tissue_names_official[tissue_id])
        tissue_counts[tissue_name_safe] = tissue_counts.get(tissue_name_safe, 0) + 1
        idx = tissue_counts[tissue_name_safe] - 1
        fname = f"{tissue_name_safe}_{idx:04d}.png"
        
        # Save image
        sample["image"].save(out_img_dir / fname)
        
        # Convert instances list of PIL bool images -> (N, 256, 256) bool array
        inst_list = sample["instances"]  # List of PIL Image mode='1'
        if len(inst_list) > 0:
            instances_np = np.stack([np.array(img, dtype=bool) for img in inst_list])
        else:
            instances_np = np.zeros((0, 256, 256), dtype=bool)
        
        # Categories list (0-4) - 0-indexed: 0=Neoplastic, 1=Inflam, ...
        categories_np = np.array(sample["categories"], dtype=np.int64)
        
        np.save(out_gt_dir / f"{tissue_name_safe}_{idx:04d}_instances.npy",
                instances_np)
        np.save(out_gt_dir / f"{tissue_name_safe}_{idx:04d}_categories.npy",
                categories_np)
        
        metadata.append({
            "filename": fname,
            "tissue_id": tissue_id,
            "tissue_name": tissue_names_official[tissue_id],
            "tissue_safe": tissue_name_safe,
            "n_nuclei": len(inst_list),
        })
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(ds)}] {elapsed:.0f}s elapsed ({(i+1)/elapsed:.1f} img/s)")
    
    with open(out_img_dir / "metadata.json", "w") as f:
        json.dump({"metadata": metadata,
                   "tissue_names_official": tissue_names_official,
                   "class_names_official": class_names_official}, f)
    print(f"  Saved in {time.time()-t0:.0f}s")

# ──────────── Statistics on the dataset ────────────
print(f"\n=== Per-tissue distribution (fold 3) ===")
t_counts = {n: 0 for n in tissue_names_official}
for sample in ds:
    t_counts[tissue_names_official[int(sample["tissue"])]] += 1
for n in tissue_names_official:
    print(f"  {n:<18}: {t_counts[n]:>4}")
print(f"  {'TOTAL':<18}: {sum(t_counts.values()):>4}")

print(f"\n=== Per-class distribution (fold 3, all 19 tissues) ===")
class_totals = {0:0, 1:0, 2:0, 3:0, 4:0}
imgs_with = {0:0, 1:0, 2:0, 3:0, 4:0}
for sample in ds:
    cats = sample["categories"]
    if not cats:
        continue
    cats_arr = np.array(cats)
    for c in range(5):
        n = int((cats_arr == c).sum())
        class_totals[c] += n
        if n > 0:
            imgs_with[c] += 1

print(f"  {'Class':<14} {'Total nuclei':>15} {'Images with class':>20}")
for c, name in enumerate(class_names_official):
    print(f"  {name:<14} {class_totals[c]:>15,d} {imgs_with[c]:>20,d}")
print(f"  {'TOTAL':<14} {sum(class_totals.values()):>15,d}")

# Symlink for convenience
proj_link = Path("./images_full_fold3")
if not proj_link.is_symlink():
    if proj_link.exists():
        import shutil
        shutil.rmtree(proj_link) if proj_link.is_dir() else proj_link.unlink()
    proj_link.symlink_to(out_img_dir)
    print(f"\n  Symlinked ./images_full_fold3 -> {out_img_dir}")
else:
    print(f"\n  Symlink already exists: {proj_link} -> {proj_link.resolve()}")
