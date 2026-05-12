"""SAM-H smoke test on GPU: load weights, forward pass, post-process, memory profile."""

import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CELLVIT_DIR = Path("./CellViT").resolve()
sys.path.insert(0, str(CELLVIT_DIR))

import torch
from PIL import Image
import torchvision.transforms as T
from models.segmentation.cell_segmentation.cellvit import CellViTSAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Device: {device} ===\n")

# ──────────── Load model ────────────
print("=== 1. Instantiating CellViT-SAM-H ===")
t0 = time.time()
model = CellViTSAM(
    model_path="/dev/null",  # not loaded; we overwrite via state_dict
    num_nuclei_classes=6,
    num_tissue_classes=19,
    vit_structure="SAM-H",
)
print(f"  Instantiated in {time.time()-t0:.1f}s")
n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params/1e6:.1f}M")
print(f"  fp32 weight size: {n_params * 4 / 1e9:.2f} GB")

# ──────────── Load weights ────────────
print("\n=== 2. Loading PanNuke pretrained weights ===")
t0 = time.time()
ckpt = torch.load("./models/cellvit_sam_h_pannuke.pth",
                  map_location="cpu", weights_only=False)
print(f"  Checkpoint loaded in {time.time()-t0:.1f}s")
print(f"  Checkpoint type: {type(ckpt).__name__}")
print(f"  Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}")

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
    print(f"  Using ckpt['model_state_dict']: {len(sd)} keys")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, arch: {ckpt.get('arch', '?')}")
else:
    sd = ckpt
    print(f"  Using ckpt directly: {len(sd)} keys")

res = model.load_state_dict(sd, strict=False)
print(f"  Missing keys:    {len(res.missing_keys)}")
print(f"  Unexpected keys: {len(res.unexpected_keys)}")
if len(res.missing_keys) > 0:
    print(f"    First 5 missing: {res.missing_keys[:5]}")
if len(res.unexpected_keys) > 0:
    print(f"    First 5 unexpected: {res.unexpected_keys[:5]}")

# ──────────── Move to GPU ────────────
print("\n=== 3. Moving to GPU ===")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = model.to(device)
model.eval()
print(f"  Moved in {time.time()-t0:.1f}s")
print(f"  GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ──────────── Preprocess image ────────────
print("\n=== 4. Preprocessing breast_00.png ===")
img_pil = Image.open("./images/breast_00.png").convert("RGB")
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
img_tensor = transform(img_pil).unsqueeze(0).to(device)
print(f"  Tensor: shape={img_tensor.shape}, dtype={img_tensor.dtype}, range=[{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

# ──────────── Forward pass — fp32 ────────────
print("\n=== 5. Forward pass (fp32) ===")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    output = model(img_tensor)
torch.cuda.synchronize()
fp32_time = time.time() - t0
fp32_peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {fp32_time:.2f}s")
print(f"  Peak GPU memory: {fp32_peak:.2f} GB")
print(f"  Output keys: {list(output.keys())}")
for k, v in output.items():
    if hasattr(v, 'shape'):
        print(f"    {k}: {tuple(v.shape)} {v.dtype}")

# ──────────── Forward pass — fp16 autocast ────────────
print("\n=== 6. Forward pass (fp16 autocast) ===")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    output_fp16 = model(img_tensor)
torch.cuda.synchronize()
fp16_time = time.time() - t0
fp16_peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  Time: {fp16_time:.2f}s")
print(f"  Peak GPU memory: {fp16_peak:.2f} GB")
print(f"  Speedup vs fp32: {fp32_time/fp16_time:.1f}x")

# ──────────── Post-processing ────────────
print("\n=== 7. Post-processing (calculate_instance_map) ===")
t0 = time.time()
# Need fp32 for post-processing (uses scipy/skimage on CPU)
output_fp32 = {k: v.float() if hasattr(v, 'float') else v for k, v in output_fp16.items()}
inst_pred, type_pred = model.calculate_instance_map(output_fp32, magnification=40)
print(f"  Time: {time.time()-t0:.1f}s")
print(f"  inst_pred: {inst_pred.shape if hasattr(inst_pred, 'shape') else type(inst_pred)}")
print(f"  Detected nuclei: {len(type_pred[0]) if type_pred else 0}")

if type_pred and len(type_pred[0]) > 0:
    nuclei = type_pred[0]
    from collections import Counter
    classes = Counter(d['type'] for d in nuclei.values())
    print(f"  Class distribution:")
    NAMES = {1:'Neoplastic', 2:'Inflammatory', 3:'Connective', 4:'Dead', 5:'Epithelial'}
    for c, n in sorted(classes.items()):
        print(f"    {NAMES.get(c, c):<14}: {n}")

# ──────────── Throughput estimate ────────────
print("\n=== 8. Throughput estimate (10 timed iterations) ===")
torch.cuda.empty_cache()
times = []
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    # warm-up
    _ = model(img_tensor)
    torch.cuda.synchronize()
    for i in range(10):
        t0 = time.time()
        _ = model(img_tensor)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

import statistics
mean_t = statistics.mean(times)
print(f"  Mean fp16 forward: {mean_t*1000:.1f} ms ({1/mean_t:.1f} img/s)")
print(f"  Estimated time for 2,600 fold-3 images:")
print(f"    forward only:  {2600 * mean_t / 60:.1f} min")
print(f"    + post-proc:   ~{2600 * (mean_t + 1.5) / 60:.0f} min  (post-proc ~1.5s/img)")

print("\n=== DONE ===")
