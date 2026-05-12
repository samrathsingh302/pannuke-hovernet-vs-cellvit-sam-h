"""Inspect HoVer-Net output to understand the data structure."""

import joblib
import numpy as np
from collections import Counter

# TIAToolbox saves outputs as joblib-pickled dicts
output_path = "./outputs/smoke_test/0.dat"
data = joblib.load(output_path)

print("=== Output type ===")
print(type(data))
print()

if isinstance(data, dict):
    print(f"=== Top-level keys ({len(data)} entries) ===")
    sample_key = list(data.keys())[0]
    print(f"Sample key: {sample_key} (type: {type(sample_key)})")
    print(f"Total nuclei detected: {len(data)}")
    print()
    
    print("=== One nucleus's data structure ===")
    sample_nucleus = data[sample_key]
    print(f"Type: {type(sample_nucleus)}")
    if isinstance(sample_nucleus, dict):
        for k, v in sample_nucleus.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: array shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
    print()
    
    # Class distribution if 'type' is available
    if isinstance(sample_nucleus, dict) and 'type' in sample_nucleus:
        type_counts = Counter(d['type'] for d in data.values())
        print("=== Class distribution ===")
        # PanNuke class mapping
        class_names = {0: 'Background', 1: 'Neoplastic', 2: 'Inflammatory',
                       3: 'Connective', 4: 'Dead', 5: 'Epithelial'}
        for type_id, count in sorted(type_counts.items()):
            name = class_names.get(type_id, f'Unknown({type_id})')
            print(f"  Class {type_id} ({name}): {count} nuclei")
