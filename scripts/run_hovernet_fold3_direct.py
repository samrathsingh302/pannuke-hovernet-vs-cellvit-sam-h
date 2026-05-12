"""HoVer-Net direct PyTorch inference. Pipeline:
- Pad 256 → 352 reflection (256 + 2*48)
- Batch 32 GPU forward via HoVerNet.infer_batch → 260x260 output
- Center crop output to 256x256 (align with GT)
- 12-worker multiprocessing pool for CPU-bound watershed postproc
Expected: ~3-5 min total for 2,722 images.
"""

import os, sys, glob, time, shutil
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import joblib
from PIL import Image

from tiatoolbox.models.architecture.hovernet import HoVerNet


# Module-level so multiprocessing can pickle
OUT_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/outputs_full/hovernet_fold3")

def postproc_and_save(per_img_maps, name):
    pred_inst, inst_info = HoVerNet.postproc(per_img_maps)
    result = {
        "nuclei": inst_info,
        "instance_map": pred_inst.astype(np.int32),
        "image_name": name,
        "n_nuclei": len(inst_info),
    }
    joblib.dump(result, OUT_DIR / f"{name}.dat", compress=0)
    return name


def main():
    device = torch.device("cuda")
    print(f"PyTorch: {torch.__version__}, device: {device}", flush=True)
    
    # Load model
    print("\n=== Loading HoVer-Net ===", flush=True)
    model = HoVerNet(num_input_channels=3, num_types=6, mode="fast")
    sd = torch.load("./models/tiatoolbox_weights/hovernet_fast-pannuke.pth",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    
    # Sanity check: 352 input → 260 output
    print("\n=== Sanity check 352→260 ===", flush=True)
    test = torch.randint(0, 256, (1, 352, 352, 3), dtype=torch.uint8)
    with torch.no_grad():
        o = HoVerNet.infer_batch(model, test, device='cuda')
    if o[0].shape[1] == 260:
        print(f"  ✅ 352×352 → 260×260 (will crop to 256×256)")
    else:
        print(f"  ❌ Got {o[0].shape[1]} not 260, aborting"); sys.exit(1)
    
    # Setup paths
    IMG_DIR = Path("/local/data/sc23sp2/dissertation/heavy_data/images_full_fold3")
    if OUT_DIR.exists():
        print(f"\nRemoving existing {OUT_DIR}", flush=True)
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    
    image_paths = sorted(glob.glob(str(IMG_DIR / "*.png")))
    print(f"\n=== Inference on {len(image_paths)} images ===", flush=True)
    
    BATCH_SIZE = 16
    N_WORKERS = 12
    PAD = 48           # 256 + 2*48 = 352
    CROP_OFFSET = 2    # output 260 → [2:258, 2:258] = 256
    
    print(f"  Batch={BATCH_SIZE}, postproc_workers={N_WORKERS}", flush=True)
    print(f"  Pad 256→352, crop output 260→256", flush=True)
    
    mp_ctx = mp.get_context('fork')
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as pool:
        futures = []
        n_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx, i in enumerate(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            
            # Load batch (B, 256, 256, 3) uint8
            batch_imgs = np.stack([
                np.array(Image.open(p).convert("RGB")) for p in batch_paths
            ])
            
            # Pad with reflection (B, C, 352, 352) then back to (B, 352, 352, C)
            bt = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2).float()
            bt_padded = F.pad(bt, (PAD,)*4, mode='reflect')
            bt_padded = bt_padded.permute(0, 2, 3, 1).contiguous()
            
            # GPU forward → (B, 260, 260, C) for each output
            raw = HoVerNet.infer_batch(model, bt_padded, device='cuda')
            torch.cuda.empty_cache()
            
            # Center crop each output map: 260 → 256
            raw_cropped = [
                arr[:, CROP_OFFSET:CROP_OFFSET+256, CROP_OFFSET:CROP_OFFSET+256, :]
                for arr in raw
            ]
            
            # Submit per-image postproc to pool
            for j, path in enumerate(batch_paths):
                per_img = [raw_cropped[k][j] for k in range(len(raw_cropped))]
                name = Path(path).stem
                futures.append(pool.submit(postproc_and_save, per_img, name))
            
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                sub = i + len(batch_paths)
                rate = sub / elapsed
                eta = (len(image_paths) - sub) / rate / 60 if rate > 0 else 0
                print(f"  Batch {batch_idx+1}/{n_batches}: {sub}/{len(image_paths)}, "
                      f"{elapsed:.0f}s, {rate:.1f} img/s queued, ETA fwd {eta:.1f}m",
                      flush=True)
        
        forward_time = time.time() - t0
        print(f"\n=== All forward done in {forward_time:.0f}s; "
              f"waiting for {len(futures)} postproc tasks ===", flush=True)
        
        n_completed = 0
        for future in as_completed(futures):
            try:
                future.result()
                n_completed += 1
                if n_completed % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  Postproc: {n_completed}/{len(futures)} ({elapsed:.0f}s total)",
                          flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
    
    elapsed = time.time() - t0
    print(f"\n=== HOVER-NET DONE in {elapsed/60:.1f} min "
          f"({elapsed/len(image_paths)*1000:.0f}ms/img) ===", flush=True)
    
    # Stats
    files = list(OUT_DIR.glob("*.dat"))
    print(f"\n  Files written: {len(files)}/{len(image_paths)}", flush=True)
    
    total_nuc, n_ok = 0, 0
    class_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
    for f in files:
        try:
            d = joblib.load(f)
            for info in d.get('nuclei', {}).values():
                t = info.get('type', 0)
                if t in class_counts:
                    class_counts[t] += 1
                total_nuc += 1
            n_ok += 1
        except Exception:
            pass
    print(f"  Files OK: {n_ok}/{len(files)}", flush=True)
    print(f"  Total nuclei: {total_nuc:,d}", flush=True)
    NAMES = {1:'Neoplastic', 2:'Inflammatory', 3:'Connective', 4:'Dead', 5:'Epithelial'}
    for c, n in class_counts.items():
        print(f"    {NAMES[c]:<14}: {n:>6,d}", flush=True)


if __name__ == "__main__":
    main()
