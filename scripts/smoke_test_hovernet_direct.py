"""Verify TIAToolbox's HoVerNet class can be used directly without engine overhead."""

import sys, time, glob, traceback
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image

print("=== 1. Importing HoVerNet ===", flush=True)
from tiatoolbox.models.architecture.hovernet import HoVerNet
print(f"  Class: {HoVerNet}")

print("\n=== 2. Inspect __init__ signature ===")
import inspect
print(f"  {inspect.signature(HoVerNet.__init__)}")

print("\n=== 3. Available methods ===")
methods = [m for m in dir(HoVerNet) if not m.startswith("_")]
print(f"  {methods}")

print("\n=== 4. Instantiate model ===")
# HoVer-Net "fast" variant for PanNuke uses 6 types (5 classes + bg)
model = HoVerNet(num_input_channels=3, num_types=6, mode="fast")
print(f"  Created: {type(model).__name__}")

print("\n=== 5. Load weights ===")
sd = torch.load("./models/tiatoolbox_weights/hovernet_fast-pannuke.pth",
               map_location="cpu", weights_only=False)
res = model.load_state_dict(sd, strict=True)
print(f"  Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
model = model.cuda().eval()
print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n=== 6. Load 5 test images ===")
IMG_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/images_full_fold3")
test_paths = sorted(glob.glob(str(IMG_DIR / "*.png")))[:5]
print(f"  Files: {[Path(p).name for p in test_paths]}")

batch_imgs = np.stack([np.array(Image.open(p).convert("RGB")) for p in test_paths])
print(f"  Batch shape (numpy uint8): {batch_imgs.shape}")

print("\n=== 7. Forward pass via infer_batch ===")
torch.cuda.synchronize()
t0 = time.time()
try:
    raw_output = HoVerNet.infer_batch(model, batch_imgs, on_gpu=True)
    torch.cuda.synchronize()
    print(f"  Time: {time.time()-t0:.3f}s for batch of 5")
    print(f"  Output type: {type(raw_output).__name__}")
    if isinstance(raw_output, (list, tuple)):
        print(f"  Output is list/tuple of {len(raw_output)} elements:")
        for i, v in enumerate(raw_output):
            print(f"    [{i}]: type={type(v).__name__}, "
                  f"shape={getattr(v, 'shape', '?')}, "
                  f"dtype={getattr(v, 'dtype', '?')}")
    elif isinstance(raw_output, dict):
        print(f"  Output keys: {list(raw_output.keys())}")
        for k, v in raw_output.items():
            print(f"    {k}: shape={getattr(v, 'shape', '?')}")
except Exception as e:
    print(f"  ❌ infer_batch failed: {e}")
    traceback.print_exc()
    print("\n  Trying direct forward() instead...")
    batch_tensor = torch.from_numpy(batch_imgs).permute(0,3,1,2).float().cuda() / 255.0
    with torch.no_grad():
        out = model(batch_tensor)
    print(f"  forward() output: type={type(out).__name__}")
    if isinstance(out, dict):
        for k, v in out.items():
            print(f"    {k}: shape={v.shape}")

print("\n=== 8. Try postproc on first image ===")
try:
    if isinstance(raw_output, (list, tuple)):
        # Standard format: each element is (B, H, W, ...) - take first sample
        per_image_maps = [o[0] for o in raw_output]
        print(f"  Per-image maps: {[(getattr(m, 'shape', '?'), getattr(m, 'dtype', '?')) for m in per_image_maps]}")
        pred_inst, inst_info = HoVerNet.postproc(per_image_maps)
        print(f"  ✅ postproc OK")
        print(f"  pred_inst: shape={pred_inst.shape}, dtype={pred_inst.dtype}, max={pred_inst.max()}")
        print(f"  inst_info: {len(inst_info)} nuclei detected")
        if len(inst_info) > 0:
            first_id = list(inst_info.keys())[0]
            first_dict = inst_info[first_id]
            print(f"  First nucleus keys: {list(first_dict.keys())}")
except Exception as e:
    print(f"  ❌ postproc failed: {e}")
    traceback.print_exc()

print("\n=== DONE ===")
