"""HoVer-Net inference on full PanNuke fold 3 (2,722 images, GPU, 16 post-proc workers)."""

import shutil, time, glob, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

IMG_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/images_full_fold3")
OUT_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/outputs_full/hovernet_fold3")
WEIGHTS = "./models/tiatoolbox_weights/hovernet_fast-pannuke.pth"

# Clean output dir (TIAToolbox refuses to overwrite)
if OUT_DIR.exists():
    print(f"Removing existing {OUT_DIR}...", flush=True)
    shutil.rmtree(OUT_DIR)
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)

image_paths = sorted(glob.glob(str(IMG_DIR / "*.png")))
print(f"\n=== HoVer-Net inference on {len(image_paths)} images ===", flush=True)

# Build idx → filename map (TIAToolbox saves to numbered dirs)
idx_map = {i: Path(p).stem for i, p in enumerate(image_paths)}

print("Initialising HoVer-Net (PanNuke pretrained)...", flush=True)
inst_segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    pretrained_weights=WEIGHTS,
    num_loader_workers=2,
    num_postproc_workers=16,   # ⬅ 16 of 20 logical CPUs
    batch_size=16,
)

print("Running inference...", flush=True)
t0 = time.time()
output = inst_segmentor.predict(
    image_paths,
    save_dir=str(OUT_DIR),
    mode="tile",
    crash_on_exception=False,
)
elapsed = time.time() - t0
print(f"\n=== HOVER-NET COMPLETE in {elapsed/60:.1f} min ({elapsed/len(image_paths):.2f}s/img) ===", flush=True)

# Save the idx → filename map for metric computation later
with open(OUT_DIR.parent / "hovernet_idx_map.json", "w") as f:
    json.dump(idx_map, f)
print(f"  Index map: {OUT_DIR.parent / 'hovernet_idx_map.json'}", flush=True)

# Quick stats
import joblib
total, n_ok = 0, 0
for idx in range(len(image_paths)):
    dat_file = OUT_DIR / str(idx) / "0.dat"
    if dat_file.exists():
        try:
            d = joblib.load(dat_file)
            total += len(d)
            n_ok += 1
        except Exception:
            pass
print(f"  {n_ok}/{len(image_paths)} files written, {total:,d} total nuclei detected", flush=True)
