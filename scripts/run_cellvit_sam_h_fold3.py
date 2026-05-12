"""CellViT-SAM-H inference on full PanNuke fold 3 (GPU, fp16, with resume)."""

import os, sys, glob, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CELLVIT_DIR = Path("./CellViT").resolve()
sys.path.insert(0, str(CELLVIT_DIR))

import torch
import joblib
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.segmentation.cell_segmentation.cellvit import CellViTSAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Device: {device} ===", flush=True)

# ──────────── Load model ────────────
print("Loading CellViT-SAM-H architecture...", flush=True)
model = CellViTSAM(
    model_path="/dev/null",          # not loaded; we override via state_dict
    num_nuclei_classes=6,
    num_tissue_classes=19,
    vit_structure="SAM-H",
)
print("Loading PanNuke pretrained weights...", flush=True)
ckpt = torch.load("./models/cellvit_sam_h_pannuke.pth", map_location="cpu", weights_only=False)
res = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"  Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}", flush=True)
print(f"  Checkpoint epoch: {ckpt.get('epoch')}, arch: {ckpt.get('arch')}", flush=True)

model = model.to(device).eval()
print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ──────────── Paths ────────────
IMG_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/images_full_fold3")
OUT_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/outputs_full/cellvit_sam_h_fold3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(glob.glob(str(IMG_DIR / "*.png")))

# Resume capability: skip already-processed images
already_done = set(p.stem for p in OUT_DIR.glob("*.dat"))
remaining = [p for p in image_paths if Path(p).stem not in already_done]
print(f"  {len(image_paths)} total, {len(already_done)} already done, {len(remaining)} remaining", flush=True)

# ──────────── Inference loop ────────────
t0 = time.time()
for i, img_path in enumerate(remaining):
    name = Path(img_path).stem
    
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x)
    
    # fp32 for post-processing (uses scipy/skimage on CPU)
    output_fp32 = {k: v.float() if hasattr(v, 'float') else v for k, v in output.items()}
    inst_pred, type_pred = model.calculate_instance_map(output_fp32, magnification=40)
    
    nuclei = type_pred[0] if (type_pred and len(type_pred) > 0) else {}
    tissue_idx = int(output_fp32["tissue_types"].argmax(dim=1).item())
    
    result = {
        "nuclei": nuclei,
        "instance_map": inst_pred.cpu().numpy()[0] if hasattr(inst_pred, 'cpu') else inst_pred,
        "tissue_pred_idx": tissue_idx,
        "image_name": name,
        "n_nuclei": len(nuclei),
    }
    joblib.dump(result, OUT_DIR / f"{name}.dat", compress=3)
    
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta_min = (len(remaining) - i - 1) / rate / 60
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  [{i+1}/{len(remaining)}] {elapsed:.0f}s, "
              f"{rate:.2f} img/s, ETA {eta_min:.0f} min, GPU {gpu_mem:.1f}GB", flush=True)

print(f"\n=== SAM-H COMPLETE in {(time.time()-t0)/60:.1f} min ===", flush=True)
print(f"  Files written: {len(list(OUT_DIR.glob('*.dat')))}", flush=True)
