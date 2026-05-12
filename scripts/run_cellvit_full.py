"""Run CellViT-256 on all 50 PanNuke test images, save in HoVer-Net-comparable format."""

import sys
import time
import shutil
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add CellViT to path
CELLVIT_DIR = Path("./CellViT").resolve()
sys.path.insert(0, str(CELLVIT_DIR))

import torch
import joblib
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.segmentation.cell_segmentation.cellvit import CellViT

# Output dir
out_dir = Path("./outputs/cellvit_all")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)

# Load model
print("=== Loading CellViT-256 model ===")
ckpt = torch.load("./models/cellvit_256_pannuke.pth",
                  map_location="cpu", weights_only=False)
config = ckpt["config"]
model = CellViT(
    num_nuclei_classes=config.get('data.num_nuclei_classes', 6),
    num_tissue_classes=config.get('data.num_tissue_classes', 19),
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
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
print(f"  Loaded weights from epoch {ckpt['epoch']}")

# Standard preprocessing
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Process all images
image_paths = sorted(glob.glob("./images/*.png"))
print(f"\n=== Running inference on {len(image_paths)} images ===\n")

t0 = time.time()
total_nuclei = 0
class_totals = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

for i, img_path in enumerate(image_paths):
    img_name = Path(img_path).stem  # e.g. "breast_00"
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        inst_pred, type_pred = model.calculate_instance_map(output, magnification=40)
    
    # type_pred is a list of dicts (one per batch item)
    nuclei_dict = type_pred[0] if type_pred else {}
    
    # Get tissue prediction
    tissue_idx = output["tissue_types"][0].argmax().item()
    
    # Save in HoVer-Net-comparable format
    save_data = {
        "nuclei": nuclei_dict,  # {id: {bbox, centroid, contour, type_prob, type}}
        "instance_map": inst_pred[0].cpu().numpy().astype(np.int32),
        "tissue_pred_idx": tissue_idx,
        "image_name": img_name,
    }
    joblib.dump(save_data, out_dir / f"{img_name}.dat", compress=3)
    
    n = len(nuclei_dict)
    total_nuclei += n
    for nid, info in nuclei_dict.items():
        cls = info.get("type", 0)
        if cls in class_totals:
            class_totals[cls] += 1
    
    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(image_paths) - i - 1) / rate
        print(f"  [{i+1:2d}/{len(image_paths)}] {img_name}: {n:3d} nuclei | {rate:.1f} img/s | ETA {eta:.0f}s")

elapsed = time.time() - t0
print(f"\n=== INFERENCE COMPLETE in {elapsed:.1f}s ({elapsed/len(image_paths):.2f}s/image) ===")
print(f"\nTotal nuclei detected across {len(image_paths)} images: {total_nuclei}")
print(f"Average per image: {total_nuclei / len(image_paths):.1f}")

class_names = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'}
print(f"\nClass totals (CellViT, all 50 images):")
for cid, count in class_totals.items():
    print(f"  {class_names[cid]:<15}: {count}")
