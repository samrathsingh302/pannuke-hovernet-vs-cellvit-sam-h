"""Generate per-tissue, per-class quantitative comparison tables."""

import joblib
from collections import defaultdict
from pathlib import Path

# Load all stats
all_stats = joblib.load("./outputs/comparison/all_stats.dat")
print(f"Loaded {len(all_stats)} image stats\n")

# PanNuke class names
CLASS_NAMES = {1: 'Neoplastic', 2: 'Inflammatory', 3: 'Connective', 4: 'Dead', 5: 'Epithelial'}

# ---------- Aggregate per-tissue ----------
per_tissue = defaultdict(lambda: {
    'gt_total': 0, 'hn_total': 0, 'cv_total': 0, 'count': 0,
    'gt_classes': {c: 0 for c in CLASS_NAMES},
    'hn_classes': {c: 0 for c in CLASS_NAMES},
    'cv_classes': {c: 0 for c in CLASS_NAMES},
})

for s in all_stats:
    t = s['tissue']
    per_tissue[t]['gt_total'] += s['gt']
    per_tissue[t]['hn_total'] += s['hn']
    per_tissue[t]['cv_total'] += s['cv']
    per_tissue[t]['count'] += 1
    for c in CLASS_NAMES:
        per_tissue[t]['gt_classes'][c] += s['gt_classes'].get(c, 0)
        per_tissue[t]['hn_classes'][c] += s['hn_classes'].get(c, 0)
        per_tissue[t]['cv_classes'][c] += s['cv_classes'].get(c, 0)

# ---------- TABLE 1: Detection counts per tissue ----------
print("="*80)
print("TABLE 1: Detection Counts Per Tissue (totals across 10 images each)")
print("="*80)
print(f"\n{'Tissue':<10} | {'GT':>6} | {'HoVer-Net':>10} | {'CellViT':>9} | {'HN%GT':>8} | {'CV%GT':>8}")
print("-"*70)
for tissue in sorted(per_tissue):
    p = per_tissue[tissue]
    hn_pct = 100 * p['hn_total'] / p['gt_total'] if p['gt_total'] else 0
    cv_pct = 100 * p['cv_total'] / p['gt_total'] if p['gt_total'] else 0
    print(f"{tissue:<10} | {p['gt_total']:>6} | {p['hn_total']:>10} | {p['cv_total']:>9} | {hn_pct:>7.1f}% | {cv_pct:>7.1f}%")
total_gt = sum(p['gt_total'] for p in per_tissue.values())
total_hn = sum(p['hn_total'] for p in per_tissue.values())
total_cv = sum(p['cv_total'] for p in per_tissue.values())
print("-"*70)
print(f"{'OVERALL':<10} | {total_gt:>6} | {total_hn:>10} | {total_cv:>9} | "
      f"{100*total_hn/total_gt:>7.1f}% | {100*total_cv/total_gt:>7.1f}%")
print()

# ---------- TABLE 2: Class distribution overall ----------
print("="*80)
print("TABLE 2: Class Distribution (across all 50 images)")
print("="*80)
gt_total_cls = {c: sum(per_tissue[t]['gt_classes'][c] for t in per_tissue) for c in CLASS_NAMES}
hn_total_cls = {c: sum(per_tissue[t]['hn_classes'][c] for t in per_tissue) for c in CLASS_NAMES}
cv_total_cls = {c: sum(per_tissue[t]['cv_classes'][c] for t in per_tissue) for c in CLASS_NAMES}

print(f"\n{'Class':<14} | {'GT':>6} | {'HoVer-Net':>10} | {'CellViT':>9} | {'HN%GT':>8} | {'CV%GT':>8}")
print("-"*70)
for c, name in CLASS_NAMES.items():
    g, h, v = gt_total_cls[c], hn_total_cls[c], cv_total_cls[c]
    h_pct = 100 * h / g if g else 0
    v_pct = 100 * v / g if g else 0
    print(f"{name:<14} | {g:>6} | {h:>10} | {v:>9} | {h_pct:>7.1f}% | {v_pct:>7.1f}%")
print()

# ---------- TABLE 3: Per-tissue × per-class (compact) ----------
print("="*80)
print("TABLE 3: Per-Tissue Class Distribution — GROUND TRUTH")
print("="*80)
print(f"\n{'Tissue':<10} | " + " | ".join(f"{n:>12}" for n in CLASS_NAMES.values()))
print("-"*90)
for t in sorted(per_tissue):
    cs = per_tissue[t]['gt_classes']
    print(f"{t:<10} | " + " | ".join(f"{cs[c]:>12}" for c in CLASS_NAMES))

print(f"\n{'Tissue':<10} | " + " | ".join(f"{n:>12}" for n in CLASS_NAMES.values()))
print("-"*90)
print("HoVer-Net per-tissue class distribution:")
print("-"*90)
for t in sorted(per_tissue):
    cs = per_tissue[t]['hn_classes']
    print(f"{t:<10} | " + " | ".join(f"{cs[c]:>12}" for c in CLASS_NAMES))

print(f"\n{'Tissue':<10} | " + " | ".join(f"{n:>12}" for n in CLASS_NAMES.values()))
print("-"*90)
print("CellViT per-tissue class distribution:")
print("-"*90)
for t in sorted(per_tissue):
    cs = per_tissue[t]['cv_classes']
    print(f"{t:<10} | " + " | ".join(f"{cs[c]:>12}" for c in CLASS_NAMES))
print()

# ---------- TABLE 4: Speed ----------
print("="*80)
print("TABLE 4: Inference Speed (CPU, 50 images)")
print("="*80)
print(f"\n{'Model':<12} | {'Total time':>12} | {'Per image':>12} | {'Speed':>10}")
print("-"*55)
print(f"{'HoVer-Net':<12} | {'131.6 s':>12} | {'2.63 s':>12} | {'1.0×':>10}")
print(f"{'CellViT':<12} | {'26.3 s':>12} | {'0.53 s':>12} | {'5.0×':>10}")
print()

# ---------- Save markdown version for dissertation ----------
md_lines = []
md_lines.append("# Comparison Results: HoVer-Net vs CellViT on PanNuke (50 images)\n")
md_lines.append("## Detection Counts Per Tissue\n")
md_lines.append("| Tissue | GT | HoVer-Net | CellViT | HN/GT | CV/GT |")
md_lines.append("|--------|---:|----------:|--------:|------:|------:|")
for t in sorted(per_tissue):
    p = per_tissue[t]
    hn_pct = 100 * p['hn_total'] / p['gt_total'] if p['gt_total'] else 0
    cv_pct = 100 * p['cv_total'] / p['gt_total'] if p['gt_total'] else 0
    md_lines.append(f"| {t} | {p['gt_total']} | {p['hn_total']} | {p['cv_total']} | {hn_pct:.1f}% | {cv_pct:.1f}% |")
md_lines.append(f"| **OVERALL** | **{total_gt}** | **{total_hn}** | **{total_cv}** | "
                f"**{100*total_hn/total_gt:.1f}%** | **{100*total_cv/total_gt:.1f}%** |")
md_lines.append("\n## Class Distribution Overall\n")
md_lines.append("| Class | GT | HoVer-Net | CellViT | HN/GT | CV/GT |")
md_lines.append("|-------|---:|----------:|--------:|------:|------:|")
for c, name in CLASS_NAMES.items():
    g, h, v = gt_total_cls[c], hn_total_cls[c], cv_total_cls[c]
    h_pct = 100 * h / g if g else 0
    v_pct = 100 * v / g if g else 0
    md_lines.append(f"| {name} | {g} | {h} | {v} | {h_pct:.1f}% | {v_pct:.1f}% |")
md_lines.append("\n## Inference Speed (CPU)\n")
md_lines.append("| Model | Total time | Per image | Relative speed |")
md_lines.append("|-------|-----------:|----------:|---------------:|")
md_lines.append("| HoVer-Net | 131.6 s | 2.63 s | 1.0× |")
md_lines.append("| CellViT | 26.3 s | 0.53 s | 5.0× faster |")

md_path = Path("./outputs/comparison/results_tables.md")
md_path.write_text("\n".join(md_lines))
print(f"\n=== Saved markdown tables to {md_path} ===")
print("    (paste this directly into your dissertation results section)")
