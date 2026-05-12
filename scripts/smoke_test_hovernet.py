"""Smoke test: Run HoVer-Net on a single breast image via TIAToolbox.

Uses LOCAL weights to bypass the broken Warwick auto-download.
TIAToolbox 1.6 auto-detects GPU/CPU - no need to specify.
"""

import shutil
from pathlib import Path
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

print("=== Setting up smoke test ===")

test_image = "./images/breast_00.png"
local_weights = "./models/tiatoolbox_weights/hovernet_fast-pannuke.pth"
out_dir = Path("./outputs/smoke_test")

if out_dir.exists():
    shutil.rmtree(out_dir)

print(f"Test image: {test_image}")
print(f"Local weights: {local_weights}")
print(f"Output dir: {out_dir}")
print()

print("=== Loading HoVer-Net (using LOCAL weights) ===")
inst_segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    pretrained_weights=local_weights,
    num_loader_workers=0,
    num_postproc_workers=0,
    batch_size=1,
)

print()
print("=== Running inference (30-60 secs on CPU) ===")
output = inst_segmentor.predict(
    [test_image],
    save_dir=str(out_dir),
    mode="tile",
    crash_on_exception=True,
)

print()
print("=== INFERENCE COMPLETE ===")
print(f"Output: {output}")
print()
print("=== Files generated ===")
for f in out_dir.rglob("*"):
    if f.is_file():
        print(f"  {f.relative_to(out_dir)} ({f.stat().st_size} bytes)")
