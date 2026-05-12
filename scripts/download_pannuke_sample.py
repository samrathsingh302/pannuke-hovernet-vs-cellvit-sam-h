"""Download a small PanNuke sample (50 images, 5 tissues) from Hugging Face."""

from pathlib import Path
from datasets import load_dataset
import numpy as np
import json
from collections import defaultdict

# PanNuke tissue mapping (alphabetical order, integer IDs)
TISSUE_NAMES = {
    0: "adrenal", 1: "bile_duct", 2: "bladder", 3: "breast", 4: "cervix",
    5: "colon", 6: "esophagus", 7: "head_neck", 8: "kidney", 9: "liver",
    10: "lung", 11: "ovarian", 12: "pancreatic", 13: "prostate", 14: "skin",
    15: "stomach", 16: "testis", 17: "thyroid", 18: "uterus"
}

# Target tissue IDs (Breast, Colon, Kidney, Liver, Lung)
TARGET_TISSUE_IDS = [3, 5, 8, 9, 10]
IMAGES_PER_TISSUE = 10

out_dir = Path("./images")
mask_dir = Path("./images/ground_truth_masks")
out_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

print("=== Loading PanNuke fold3 (cached from previous run) ===")
ds = load_dataset("RationAI/PanNuke", split="fold3")
print(f"Total images: {len(ds)}\n")

tissue_counts = defaultdict(int)
metadata = []

for i, sample in enumerate(ds):
    tissue_id = sample["tissue"]
    
    if tissue_id not in TARGET_TISSUE_IDS:
        continue
    if tissue_counts[tissue_id] >= IMAGES_PER_TISSUE:
        continue
    
    tissue_name = TISSUE_NAMES[tissue_id]
    idx = tissue_counts[tissue_id]
    filename = f"{tissue_name}_{idx:02d}"
    
    # Save image
    sample["image"].save(out_dir / f"{filename}.png")
    
    # Save ground truth masks
    inst_mask = np.array(sample["instances"])
    cat_mask = np.array(sample["categories"])
    np.save(mask_dir / f"{filename}_instances.npy", inst_mask)
    np.save(mask_dir / f"{filename}_categories.npy", cat_mask)
    
    metadata.append({
        "filename": f"{filename}.png",
        "tissue_id": int(tissue_id),
        "tissue_name": tissue_name,
        "n_instances": int(inst_mask.max()) if inst_mask.max() > 0 else 0,
        "original_idx": i
    })
    
    tissue_counts[tissue_id] += 1
    
    if all(tissue_counts[t] >= IMAGES_PER_TISSUE for t in TARGET_TISSUE_IDS):
        break

with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("=== Saved Images Per Tissue ===")
for tid, count in sorted(tissue_counts.items()):
    print(f"  {TISSUE_NAMES[tid]:12s} (id={tid}): {count}")
print(f"\nTotal: {sum(tissue_counts.values())} images")
print(f"Location: {out_dir.resolve()}")
