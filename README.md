# HoVer-Net vs CellViT-SAM-H on PanNuke fold 3

Independent benchmark of two pretrained nuclei instance segmentation models. Final-year project, University of Leeds, 2026.

## Result

On 2,722 PanNuke fold-3 images (66,654 ground-truth nuclei across 19 tissues and 5 classes), CellViT-SAM-H outperforms HoVer-Net on every aggregate metric, every tissue (19/19), and every nuclei class (5/5). HoVer-Net reproduces Graham et al. (2019) to within 0.001 bPQ and 0.012 mPQ, validating the metric pipeline.

| metric | HoVer-Net | CellViT-SAM-H | Δ |
|---|---:|---:|---:|
| bPQ | 0.6583 [0.6512, 0.6646] | 0.7963 [0.7907, 0.8019] | +0.1380 |
| mPQ | 0.4510 [0.4412, 0.4606] | 0.7048 [0.6952, 0.7142] | +0.2538 |
| Runtime (per patch) | 85 ms | 121 ms | 1.4× |
| Parameters | ~50M | 699.7M | 14× |

95% bootstrap CIs in brackets, n=1,000. Hardware: NVIDIA RTX 4070, 12 GB VRAM.

## What's in this repository
scripts/   inference, metric, validation, and figure pipelines
docs/      raw metric outputs (metrics_results_official.json, validation_report.json)
env/       pinned package versions used during evaluation
## Reproducing the result

**Hardware**: any NVIDIA GPU with ≥6 GB VRAM. Tested on RTX 4070 (12 GB).
**OS**: Linux. Tested on RHEL 9.7.
**Total disk**: ≈5 GB.
**Total runtime**: ≈10 minutes of compute (3.9 min HoVer-Net + 5.5 min CellViT + metric pass).

The scripts contain absolute paths to the original Leeds lab environment (`/local/data/sc23sp2/...` and `/uolstore/home/users/sc23sp2/...`). To reproduce on a different machine, you will need to edit `HD = Path(...)` near the top of each script to point at your local working directory.

### 1. Clone this repository
git clone https://github.com/samrathsingh302/pannuke-hovernet-vs-cellvit-sam-h.git
cd pannuke-hovernet-vs-cellvit-sam-h
### 2. Create three Python environments

The HoVer-Net, CellViT-SAM-H, and metric stages have different package pins (mainly NumPy and PyTorch). Each `env/env_*.txt` is the output of `pip freeze` from the corresponding venv.
### 2. Create three Python environments

The HoVer-Net, CellViT-SAM-H, and metric stages have different package pins (mainly NumPy and PyTorch). Each `env/env_*.txt` is the output of `pip freeze` from the corresponding venv.
python3.9  -m venv venv_hn  && source venv_hn/bin/activate  && pip install -r env/env_hovernet.txt  && deactivate
python3.10 -m venv venv_cv  && source venv_cv/bin/activate  && pip install -r env/env_cellvit.txt   && deactivate
python3.9  -m venv venv_mx  && source venv_mx/bin/activate  && pip install -r env/env_metrics.txt   && deactivate
### 3. Clone the two external repositories the pipeline depends on
git clone https://github.com/TIO-IKIM/CellViT
git clone https://github.com/TissueImageAnalytics/PanNuke-metrics
Both are imported by path in the scripts. CellViT is needed for the `CellViTSAM` model class; PanNuke-metrics is the official bPQ/mPQ implementation.

### 4. Download model weights

| File | Source | Size |
|------|--------|------|
| HoVer-Net (PanNuke) | https://tiatoolbox.dcs.warwick.ac.uk/models/seg/hovernet_fast-pannuke.pth | 145 MB |
| CellViT-SAM-H (PanNuke) | https://github.com/TIO-IKIM/CellViT → README → Models → CellViT-SAM-H Google Drive link | 2.7 GB |

The CellViT-SAM-H weights are hosted on Google Drive (the CellViT repository README links to them). Both weights are CC BY-NC-SA 4.0 — not redistributed here. Save them to wherever your scripts expect (default in the scripts: `/local/data/sc23sp2/dissertation/heavy_data/models/`; edit if running elsewhere).

### 5. Run the pipeline

Activate the right venv for each stage. The scripts must be run in order — each stage consumes the previous stage's output.
source venv_hn/bin/activate
python scripts/download_full_fold3.py
python scripts/run_hovernet_fold3_direct.py
deactivate
source venv_cv/bin/activate
python scripts/run_cellvit_sam_h_fold3.py
deactivate
source venv_mx/bin/activate
python scripts/compute_metrics_official.py
python scripts/validate_everything.py
python scripts/generate_dissertation_figures.py
deactivate
Outputs after stage 5:
- `metrics_results_official.json` — overall + per-tissue + per-class bPQ/mPQ with 95% CIs
- `validation_report.json` — 94-check validation framework results
- `dissertation_figures/*.png` — six PNG figures

## Dataset

PanNuke (Gamper et al. 2019, 2020), CC BY-NC-SA 4.0, downloaded from HuggingFace mirror `RationAI/PanNuke` config `default` split `fold3`. Not redistributed here.

## Citation

If you use this work please cite as:
Singh, S. (2026). An independent benchmark of HoVer-Net and CellViT-SAM-H
on the PanNuke dataset. COMP3931 final-year project, School of Computing,
University of Leeds.
https://github.com/samrathsingh302/pannuke-hovernet-vs-cellvit-sam-h
## License

Code: MIT (see LICENSE). Model weights and PanNuke dataset are CC BY-NC-SA 4.0 from their respective authors and are not redistributed here.

## Supervisor

Dr Arash Rabbani, School of Computing, University of Leeds.
