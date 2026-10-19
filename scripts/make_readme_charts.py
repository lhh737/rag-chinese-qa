"""Generate the two README result charts as plain SVG (no runtime deps).

    uv run python scripts/make_readme_charts.py

Outputs:
    assets/domain-quality.svg
    assets/false-refusal.svg

Design notes:
- palette: #2a78d6 / #eb6834 (validated with the dataviz palette validator)
- one axis per chart; no dual axis
- bars: 24px thick, 2px surface gap, 4px rounded data-end
- false-refusal: dot plot with Wilson 95% CI whiskers
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def text(x: float, y: float, s: str, size: int = 11, fill: str = INK2,
         anchor: str = "start", weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family=\'{FONT}\' font-variant-numeric="tabular-nums">{esc(s)}</text>')


def rounded_top_bar(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    return (f"M {x:.1f},{y + h:.1f} L {x:.1f},{y + r:.1f} "
            f"Q {x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"L {x + w - r:.1f},{y:.1f} Q {x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"L {x + w:.1f},{y + h:.1f} Z")


def domain_quality() -> str:
    W, H = 860, 400
    x0, x1, y0, y1 = 64, 836, 84, 336
    data = [
        ("Paper R1", "dev 60", 0.717, 0.692),
        ("Paper final", "dev 60", 0.758, 0.725),
        ("Paper holdout", "20", 0.800, 0.667),
        ("E-commerce FAQ", "30", 0.825, 0.692),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="dq-title dq-desc">',
        f'<title id="dq-title">Domain comparison: retrieval and answer quality</title>',
        f'<desc id="dq-desc">Grouped bars of Recall@5 and judge-scored answer correctness across four evaluation sets. Recall@5 is not directly comparable across corpora.</desc>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        text(64, 32, "Domain comparison: retrieval and answer quality", 16, INK, weight="600"),
        text(64, 52, "Recall@5 and judge-scored answer correctness. R@5 is not directly comparable across corpora (106 vs 30 documents).", 11, INK2),
        # legend
        f'<rect x="636" y="22" width="10" height="10" rx="0" fill="{BLUE}"/>',
        text(652, 31, "Recall@5", 11, INK2),
        f'<rect x="726" y="22" width="10" height="10" rx="0" fill="{ORANGE}"/>',
        text(742, 31, "Answer correctness", 11, INK2),
    ]
    # grid + y ticks
    for i in range(5):
        v = i / 4
        y = y1 - v * (y1 - y0)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(text(x0 - 10, y + 4, f"{v:.2f}", 11, MUTED, anchor="end"))
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{AXIS}" stroke-width="1"/>')

    band = (x1 - x0) / len(data)
    for i, (label, sub, r5, ac) in enumerate(data):
        cx = x0 + band * (i + 0.5)
        bw, gap = 24, 2
        start = cx - (bw * 2 + gap) / 2
        for j, (value, color, name) in enumerate(((r5, BLUE, "Recall@5"), (ac, ORANGE, "Answer correctness"))):
            h = value * (y1 - y0)
            x = start + j * (bw + gap)
            y = y1 - h
            parts.append(
                f'<path d="{rounded_top_bar(x, y, bw, h)}" fill="{color}">'
                f'<title>{esc(label)} {esc(sub)} — {name}: {value:.3f}</title></path>'
            )
        # direct labels only for the business domain (the subject of the comparison)
        if label == "E-commerce FAQ":
            parts.append(text(start + bw / 2, y1 - r5 * (y1 - y0) - 7, f"{r5:.3f}", 10, INK, anchor="middle"))
            parts.append(text(start + bw + gap + bw / 2, y1 - ac * (y1 - y0) - 7, f"{ac:.3f}", 10, INK, anchor="middle"))
        parts.append(text(cx, y1 + 20, label, 11, INK2, anchor="middle"))
        parts.append(text(cx, y1 + 35, sub, 10, MUTED, anchor="middle"))

    parts.append(text(64, 382, "Paper R1 = HyDE on; paper final = R1 + HyDE off. E-commerce FAQ uses the final config.", 10, MUTED))
    parts.append("</svg>")
    return "\n".join(parts)


def false_refusal() -> str:
    W, H = 860, 360
    x0, x1, y0, y1 = 140, 820, 84, 304
    data = [
        ("Paper dev R1", 23.1, 13.7, 36.1),
        ("Paper dev final", 23.1, 13.7, 36.1),
        ("Paper holdout", 16.7, 5.8, 39.2),
        ("E-commerce FAQ", 23.1, 11.0, 42.1),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="fr-title fr-desc">',
        '<title id="fr-title">False refusal with 95% Wilson confidence intervals</title>',
        '<desc id="fr-desc">Dot plot of the false-refusal rate across four evaluation sets, with 95% Wilson confidence intervals. Lower is better.</desc>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        text(64, 32, "False refusal with 95% Wilson CI", 16, INK, weight="600"),
        text(64, 52, "Lower is better. Paper dev n=60; paper holdout n=20; business n=30.", 11, INK2),
    ]
    # x grid + ticks (0–50%)
    for v in range(0, 51, 10):
        x = x0 + (v / 50) * (x1 - x0)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(text(x, y1 + 20, f"{v}%", 11, MUTED, anchor="middle"))
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{AXIS}" stroke-width="1"/>')

    row_h = (y1 - y0) / len(data)
    for i, (label, value, lo, hi) in enumerate(data):
        y = y0 + row_h * (i + 0.5)
        xv = x0 + (value / 50) * (x1 - x0)
        xl = x0 + (lo / 50) * (x1 - x0)
        xh = x0 + (hi / 50) * (x1 - x0)
        parts.append(text(x0 - 12, y + 4, label, 12, INK2, anchor="end"))
        parts.append(
            f'<line x1="{xl:.1f}" y1="{y:.1f}" x2="{xh:.1f}" y2="{y:.1f}" stroke="{INK2}" stroke-width="1.5"/>'
            f'<line x1="{xl:.1f}" y1="{y - 5:.1f}" x2="{xl:.1f}" y2="{y + 5:.1f}" stroke="{INK2}" stroke-width="1.5"/>'
            f'<line x1="{xh:.1f}" y1="{y - 5:.1f}" x2="{xh:.1f}" y2="{y + 5:.1f}" stroke="{INK2}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<circle cx="{xv:.1f}" cy="{y:.1f}" r="4.5" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2">'
            f'<title>{esc(label)} — {value:.1f}% (95% CI {lo:.1f}–{hi:.1f}%)</title></circle>'
        )
        parts.append(
            f'<text x="{xv:.1f}" y="{y - 10:.1f}" font-size="11" fill="{INK}" '
            f'text-anchor="middle" font-family=\'{FONT}\' font-variant-numeric="tabular-nums" '
            f'stroke="{SURFACE}" stroke-width="3" paint-order="stroke">{value:.1f}%</text>'
        )
    parts.append(text(64, 344, "Intervals are Wilson 95% CI. Holdout and business are not directly comparable to paper dev.", 10, MUTED))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "domain-quality.svg").write_text(domain_quality(), encoding="utf-8")
    (OUT / "false-refusal.svg").write_text(false_refusal(), encoding="utf-8")
    print(f"✓ {(OUT / 'domain-quality.svg').relative_to(ROOT)}")
    print(f"✓ {(OUT / 'false-refusal.svg').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
