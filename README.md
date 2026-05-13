# HoVer-Net vs CellViT-SAM-H on PanNuke fold 3

An independent benchmark of two pretrained nuclei instance segmentation model - a CNN (HoVer-Net) and a Vision Transformer (CellViT-SAM-H) - on the held-out third fold of the PanNuke dataset. Final-year project, School of Computing, University of Leeds, 2026.

## Headline result

CellViT-SAM-H outperforms HoVer-Net on every aggregate metric. This is for all 2,722 PanNuke fold-3 images (66,654 ground-truth nuclei across 19 tissues and 5 classes, and every individual tissue (19 of 19) and every nuclei class (5 of 5). The 95% bootstrap confidence intervals do not overlap on either metric. The HoVer-Net evaluation independently reproduces Graham et al. (2019) to within 0.001 bPQ and 0.012 mPQ, confirming the metric pipeline.

| Metric | HoVer-Net | CellViT-SAM-H | Delta |
| :--- | ---: | ---: | ---: |
| bPQ | 0.6583 [0.6512, 0.6646] | 0.7963 [0.7907, 0.8019] | +0.1380 |
| mPQ | 0.4510 [0.4412, 0.4606] | 0.7048 [0.6952, 0.7142] | +0.2538 |
| Runtime per patch | 85 ms | 121 ms | 1.4x |
| Parameters | ~50M | 699.7M | 14x |

95% bootstrap CIs shown in brackets (n = 1,000 image-level resamples). Hardware: NVIDIA RTX 4070, 12 GB VRAM.

## Repository contents

| Directory | Contents |
| :--- | :--- |
| `scripts/` | Inference, metric, validation, and figure-generation Python scripts |
| `docs/` | Raw output JSONs (metric results, validation report) |
| `env/` | Pinned package versions for the three Python environments used |

## Reproducing the result

**Hardware**: any NVIDIA GPU with at least 6 GB VRAM. Tested on RTX 4070 (12 GB).
**OS**: Linux (tested on RHEL 9.7).
**Disk**: about 5 GB total.
**Compute time**: about 10 minutes (3.9 min HoVer-Net + 5.5 min CellViT-SAM-H + metric pass).

The scripts contain absolute paths from the original Leeds lab environment. To reproduce on a different machine, edit the `HD = Path(...)` line near the top of each script to point at your local working directory.

### Step 1. Clone this repository

```bash
git clone https://github.com/samrathsingh302/pannuke-hovernet-vs-cellvit-sam-h.git
cd pannuke-hovernet-vs-cellvit-sam-h
```

### Step 2. Create three Python environments

The HoVer-Net, CellViT, and metric stages have different package pins (mainly NumPy and PyTorch versions). Each `env/env_*.txt` is the corresponding `pip freeze`.

```bash
python3.9  -m venv venv_hn  && source venv_hn/bin/activate  && pip install -r env/env_hovernet.txt  && deactivate
python3.10 -m venv venv_cv  && source venv_cv/bin/activate  && pip install -r env/env_cellvit.txt   && deactivate
python3.9  -m venv venv_mx  && source venv_mx/bin/activate  && pip install -r env/env_metrics.txt   && deactivate
```

### Step 3. Clone the two external repositories the pipeline depends on

```bash
git clone https://github.com/TIO-IKIM/CellViT
git clone https://github.com/TissueImageAnalytics/PanNuke-metrics
```

CellViT supplies the `CellViTSAM` model class. PanNuke-metrics is the official bPQ/mPQ implementation by the dataset authors.

### Step 4. Download model weights

Both checkpoints are CC BY-NC-SA 4.0 and are not redistributed here.

#### HoVer-Net (PanNuke), 145 MB

The recommended way is via the TIAToolbox API, which downloads the weights into a local cache and is robust to upstream URL changes:

```bash
source venv_hn/bin/activate
python -c "from tiatoolbox.models.architecture import fetch_pretrained_weights; fetch_pretrained_weights('hovernet_fast-pannuke', './hovernet_fast-pannuke.pth')"
deactivate
```

For an overview of all pretrained models available through TIAToolbox, see https://tia-toolbox.readthedocs.io/en/latest/pretrained.html

#### CellViT-SAM-H (PanNuke), 2.7 GB

Visit https://github.com/TIO-IKIM/CellViT and follow the "Pretrained Models" section. The CellViT-SAM-H checkpoint is hosted on Google Drive; the CellViT README provides the current link.

### Step 5. Run the pipeline

Activate the correct virtual environment for each stage. The stages must run in order: each consumes the previous stage's output.

```bash
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
```

Outputs produced:

- `metrics_results_official.json` — overall, per-tissue, and per-class bPQ/mPQ with 95% bootstrap CIs
- `validation_report.json` — 94-check validation framework results
- `dissertation_figures/*.png` — six PNG figures

## Dataset

PanNuke (Gamper et al. 2019, 2020), CC BY-NC-SA 4.0, downloaded from the HuggingFace mirror `RationAI/PanNuke`, config `default`, split `fold3`. The dataset is not redistributed here.

## Citation

```
Singh, S. (2026). An independent benchmark of HoVer-Net and CellViT-SAM-H
on the PanNuke dataset. COMP3931 final-year project,
School of Computing, University of Leeds.
https://github.com/samrathsingh302/pannuke-hovernet-vs-cellvit-sam-h
```

## License

Code: MIT (see `LICENSE`). Model weights and the PanNuke dataset are CC BY-NC-SA 4.0 from their respective authors and are not redistributed here.

## Supervisor

Dr Arash Rabbani, School of Computing, University of Leeds.
