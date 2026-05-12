"""Smoke test: Run CellViT-256 on ONE breast image."""

import sys
from pathlib import Path

# Add CellViT repo to Python path
CELLVIT_DIR = Path("./CellViT").resolve()
sys.path.insert(0, str(CELLVIT_DIR))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

print("=== Imports ===")
from models.segmentation.cell_segmentation.cellvit import CellViT, CellViT256
print("  CellViT classes imported OK")

# Load checkpoint
print("\n=== Loading checkpoint ===")
ckpt = torch.load("./models/cellvit_256_pannuke.pth",
                  map_location="cpu", weights_only=False)
config = ckpt["config"]
print(f"  arch: {ckpt['arch']}")
print(f"  epoch: {ckpt['epoch']}")

# Show key config values
print(f"\n  num_nuclei_classes: {config.get('data.num_nuclei_classes', '?')}")
print(f"  num_tissue_classes: {config.get('data.num_tissue_classes', '?')}")

n_nuc = config.get('data.num_nuclei_classes', 6)
n_tis = config.get('data.num_tissue_classes', 19)

# Instantiate using parent CellViT class (no model256_path needed)
print("\n=== Instantiating model (using parent CellViT class) ===")
model = CellViT(
    num_nuclei_classes=n_nuc,
    num_tissue_classes=n_tis,
    embed_dim=384,
    input_channels=3,
    depth=12,
    num_heads=6,
    extract_layers=[3, 6, 9, 12],
    mlp_ratio=4,
    qkv_bias=True,
    drop_rate=0,
    attn_drop_rate=0,
    drop_path_rate=0,
    regression_loss=False,
)
print(f"  Model created: {type(model).__name__}")

# Load weights
print("\n=== Loading weights ===")
res = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"  Missing keys:    {len(res.missing_keys)}")
print(f"  Unexpected keys: {len(res.unexpected_keys)}")
if res.missing_keys[:5]:
    print(f"  First 5 missing: {res.missing_keys[:5]}")
if res.unexpected_keys[:5]:
    print(f"  First 5 unexpected: {res.unexpected_keys[:5]}")

model.eval()
print("  Model in eval mode")

# Load and preprocess one image
print("\n=== Preprocessing image: breast_00.png ===")
img_pil = Image.open("./images/breast_00.png").convert("RGB")
print(f"  Image size: {img_pil.size}")

# Standard CellViT preprocessing: ImageNet stats
transform = T.Compose([
    T.ToTensor(),  # to [0,1]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # CellViT uses [-1,1] normalization
])
img_tensor = transform(img_pil).unsqueeze(0)  # (1, 3, 256, 256)
print(f"  Tensor shape: {img_tensor.shape}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

# Forward pass
print("\n=== Forward pass (CPU, ~30 sec) ===")
import time
t0 = time.time()
with torch.no_grad():
    output = model(img_tensor)
print(f"  Done in {time.time()-t0:.1f}s")

print("\n=== Output keys & shapes ===")
for k, v in output.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"  {k}: list of {len(v)} items")
    else:
        print(f"  {k}: {type(v).__name__}")

# Try post-processing via the model's built-in method
print("\n=== Attempting calculate_instance_map ===")
try:
    inst_pred, type_pred = model.calculate_instance_map(output, magnification=40)
    print(f"  inst_pred type: {type(inst_pred)}")
    if isinstance(inst_pred, torch.Tensor):
        print(f"  inst_pred shape: {inst_pred.shape}")
    elif isinstance(inst_pred, list):
        print(f"  inst_pred: list of {len(inst_pred)} items")
        if inst_pred and hasattr(inst_pred[0], 'shape'):
            print(f"  inst_pred[0] shape: {inst_pred[0].shape}")
    print(f"  type_pred type: {type(type_pred)}")
    if isinstance(type_pred, list):
        print(f"  type_pred: list of {len(type_pred)} items, first={type_pred[0] if type_pred else 'empty'}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
