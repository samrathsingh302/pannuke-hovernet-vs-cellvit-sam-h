"""Run HoVer-Net on all 50 PanNuke test images."""

import shutil
import time
import glob
from pathlib import Path
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

print("=== Setting up full HoVer-Net inference run ===")

# Gather all test images
image_paths = sorted(glob.glob("./images/*.png"))
print(f"Found {len(image_paths)} images")
for p in image_paths[:5]:
    print(f"  {p}")
print(f"  ... and {len(image_paths) - 5} more")
print()

# Output dir (clean up any previous run)
out_dir = Path("./outputs/hovernet_all")
if out_dir.exists():
    shutil.rmtree(out_dir)

local_weights = "./models/tiatoolbox_weights/hovernet_fast-pannuke.pth"

print("=== Loading HoVer-Net (PanNuke pretrained) ===")
inst_segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    pretrained_weights=local_weights,
    num_loader_workers=0,
    num_postproc_workers=0,
    batch_size=4,  # Slightly bigger for efficiency
)

print()
print(f"=== Running inference on {len(image_paths)} images ===")
print("(This takes ~4 sec/image on CPU = ~3-5 mins total)")
print()

t0 = time.time()
output = inst_segmentor.predict(
    image_paths,
    save_dir=str(out_dir),
    mode="tile",
    crash_on_exception=True,
)
elapsed = time.time() - t0

print()
print(f"=== INFERENCE COMPLETE in {elapsed:.1f}s ({elapsed/len(image_paths):.2f}s/image) ===")
print()

# Check outputs
out_files = list(out_dir.rglob("*.dat"))
print(f"Generated {len(out_files)} output files")
print()

# Quick stats
import joblib
total_nuclei = 0
for img_path, out_path in output:
    dat_file = Path(out_path).parent / (Path(out_path).name + ".dat")
    if not dat_file.exists():
        # Try common patterns
        candidates = list(Path(out_path).parent.glob("*.dat"))
        if candidates:
            dat_file = candidates[0]
    if dat_file.exists():
        data = joblib.load(dat_file)
        total_nuclei += len(data)

print(f"Total nuclei detected across {len(image_paths)} images: {total_nuclei}")
print(f"Average per image: {total_nuclei / len(image_paths):.1f}")
