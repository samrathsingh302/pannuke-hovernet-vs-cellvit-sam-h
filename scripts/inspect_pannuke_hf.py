"""Figure out the new HF dataset structure for RationAI/PanNuke."""

import os
os.environ.setdefault("HF_HOME", "/local/data/sc23sp2/dissertation/hf_cache")
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

print("=== Available configs ===")
configs = get_dataset_config_names("RationAI/PanNuke")
print(f"  configs: {configs}")

print("\n=== Splits in default config ===")
splits = get_dataset_split_names("RationAI/PanNuke", "default")
print(f"  splits: {splits}")

print("\n=== Loading default config ===")
import time
t0 = time.time()
ds = load_dataset("RationAI/PanNuke", "default")
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Type: {type(ds).__name__}")
print(f"  Splits: {list(ds.keys()) if hasattr(ds, 'keys') else 'N/A'}")

if hasattr(ds, 'keys'):
    for split_name in ds.keys():
        print(f"\n=== Split '{split_name}' ===")
        split_data = ds[split_name]
        print(f"  Total samples: {len(split_data)}")
        print(f"  Features: {split_data.features}")
        # show first sample
        if len(split_data) > 0:
            sample = split_data[0]
            for k, v in sample.items():
                if hasattr(v, 'size'):
                    print(f"    {k}: PIL Image {v.size}")
                elif isinstance(v, list):
                    arr_v = v
                    print(f"    {k}: list len={len(arr_v)}")
                else:
                    print(f"    {k}: {type(v).__name__} = {str(v)[:80]}")
        break  # just inspect first split
