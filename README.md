# PanNuke HoVer-Net vs CellViT-SAM-H Comparison

An independent benchmark of two pretrained nuclei instance segmentation models on PanNuke fold 3. COMP3931 final-year project, University of Leeds, 2026.

## Headline result

On 2,722 fold-3 images, CellViT-SAM-H outperforms HoVer-Net on every metric (Δ bPQ +0.138, Δ mPQ +0.254, 95% CIs non-overlapping), every tissue (19/19), and every nuclei class (5/5). HoVer-Net evaluation reproduces Graham et al. 2019 to within 0.012 mPQ, validating the metric pipeline.

| metric | HoVer-Net | CellViT-SAM-H | Δ |
|---|---:|---:|---:|
| bPQ | 0.6583 [0.6512, 0.6646] | 0.7963 [0.7907, 0.8019] | +0.1380 |
| mPQ | 0.4510 [0.4412, 0.4606] | 0.7048 [0.6952, 0.7142] | +0.2538 |
| Runtime | 85 ms/patch | 121 ms/patch | 1.4× |
| Parameters | ~50M | 699.7M | 14× |

## Layout

- `scripts/` — inference, metrics, validation, figure generation
- `docs/` — metrics JSON, validation report, dissertation PDF (post-submission)
- `env/` — pinned package versions

## Reproduce

```bash
python scripts/download_full_fold3.py
python scripts/run_hovernet_fold3_direct.py
python scripts/run_cellvit_sam_h_fold3.py
python scripts/compute_metrics_official.py
python scripts/validate_everything.py
python scripts/generate_dissertation_figures.py
```

Model weights not redistributed:
- HoVer-Net: https://tiatoolbox.dcs.warwick.ac.uk/models/seg/
- CellViT-SAM-H: https://github.com/TIO-IKIM/CellViT

## Supervisor

Dr Arash Rabbani, School of Computing, University of Leeds.

## License

MIT (see LICENSE). The PanNuke dataset itself is CC BY-NC-SA 4.0 and is not redistributed here.
