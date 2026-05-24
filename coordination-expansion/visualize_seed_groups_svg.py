#!/usr/bin/env python3
"""Render seed expansion groups as a lightweight SVG.

This avoids optional plotting dependencies so the report can be generated in a
minimal analysis environment.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = ["harvested", "JG87919", "BtcKing1111"]
DEFAULT_INPUT_DIR = Path("coordination-expansion/output/seeds")
DEFAULT_GRAPH_DIR = Path("adjacency/output")
DEFAULT_OUTPUT_DIR = Path("coordination-expansion/output/visuals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw seed group relationship SVG.")
    parser.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def label_lines(name: str, max_len: int = 15) -> list[str]:
    if len(name) <= max_len:
        return [name]
    parts = name.replace("_", "_ ").replace("-", "- ").split()
    lines: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(part) + 1 <= max_len:
            current += " " + part
        else:
            lines.append(current)
            current = part
    if current:
        lines.append(current)
    if len(lines) > 2:
        return [lines[0], "".join(lines[1:])[: max_len - 1] + "..."]
    return lines


def node_style(tier: int, is_seed: bool) -> tuple[str, str, int]:
    if is_seed:
        return "#172554", "#ffffff", 30
    if tier == 4:
        return "#7c3aed", "#ffffff", 23
    if tier == 1:
        return "#f97316", "#ffffff", 22
    return "#64748b", "#ffffff", 20


def edge_style(layer: str, weight: float | None) -> tuple[str, float, str]:
    if layer == "co_negative_target":
        width = 1.2 + min(max(weight or 0.0, 0.0), 0.8) * 4.0
        return "#dc2626", width, ""
    width = 1.0 + min(max(weight or 0.0, 0.0), 0.8) * 2.0
    return "#94a3b8", width, "6 5"


def draw_svg(
    *,
    seeds: list[str],
    input_dir: Path,
    graph_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    width = 1400
    height = 820
    centers = {
        seeds[0]: (260, 410),
        seeds[1]: (700, 310),
        seeds[2]: (1140, 410),
    }
    palette = {
        seeds[0]: "#ecfeff",
        seeds[1]: "#fff7ed",
        seeds[2]: "#f5f3ff",
    }

    positions: dict[str, tuple[float, float]] = {}
    tiers: dict[str, int] = {}
    group_of: dict[str, str] = {}
    group_members: dict[str, list[str]] = {}
    elements: list[str] = []

    for seed in seeds:
        members_path = input_dir / seed / "tiered_expansion_members.csv"
        members = pd.read_csv(members_path)
        members = members.loc[members["include"].eq(True)].copy()
        names = members["candidate"].astype(str).tolist()
        group_members[seed] = names
        center_x, center_y = centers[seed]
        positions[seed] = (center_x, center_y)

        for row in members.itertuples(index=False):
            candidate = str(row.candidate)
            tiers[candidate] = int(row.tier)
            group_of[candidate] = seed

        others = [name for name in names if name != seed]
        radius = 150 if len(others) <= 8 else 190
        start_angle = -math.pi / 2
        for index, name in enumerate(others):
            angle = start_angle + 2 * math.pi * index / max(len(others), 1)
            positions[name] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )

    elements.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>'
    )
    elements.append(
        '<text x="40" y="44" font-family="Arial, sans-serif" '
        'font-size="26" font-weight="700" fill="#0f172a">'
        "Seed Expansion Relationship Graph</text>"
    )
    elements.append(
        '<text x="40" y="72" font-family="Arial, sans-serif" '
        'font-size="14" fill="#475569">'
        "Solid red = co-negative target edge; dashed gray = co-target support; dashed blue = tag similarity between seed groups.</text>"
    )

    for seed in seeds:
        cx, cy = centers[seed]
        elements.append(
            f'<circle cx="{cx}" cy="{cy}" r="230" fill="{palette[seed]}" '
            'stroke="#cbd5e1" stroke-width="1.4"/>'
        )
        elements.append(
            f'<text x="{cx}" y="{cy - 245}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="18" '
            f'font-weight="700" fill="#0f172a">{esc(seed)}</text>'
        )

    drawn_edges: set[tuple[str, str, str]] = set()
    for seed in seeds:
        edge_path = input_dir / seed / "internal_coordination_edges.csv"
        edges = pd.read_csv(edge_path)
        members = set(group_members[seed])
        for row in edges.itertuples(index=False):
            source = str(row.source_author)
            target = str(row.target_author)
            if source not in members or target not in members:
                continue
            layer = str(getattr(row, "layer", ""))
            if layer not in {"co_negative_target", "co_target"}:
                continue
            key = tuple(sorted((source, target))) + (layer,)
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            weight = None
            if layer == "co_negative_target" and hasattr(row, "weight_co_negative_target"):
                value = getattr(row, "weight_co_negative_target")
                weight = None if pd.isna(value) else float(value)
            if layer == "co_target" and hasattr(row, "weight_co_target"):
                value = getattr(row, "weight_co_target")
                weight = None if pd.isna(value) else float(value)
            color, stroke_width, dash = edge_style(layer, weight)
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            elements.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" '
                f'stroke-opacity="0.72"{dash_attr}/>'
            )

    tag_path = graph_dir / "multi-graph" / "edges_tag_similarity.csv"
    if tag_path.exists():
        tag_edges = pd.read_csv(tag_path)
        seed_set = set(seeds)
        for row in tag_edges.itertuples(index=False):
            source = str(row.source_author)
            target = str(row.target_author)
            weight = float(row.weight_tag_similarity)
            if source in seed_set and target in seed_set and weight >= 0.90:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                elements.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    'stroke="#2563eb" stroke-width="4" stroke-opacity="0.85" '
                    'stroke-dasharray="10 8"/>'
                )
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                elements.append(
                    f'<rect x="{mx - 62:.1f}" y="{my - 18:.1f}" width="124" height="26" '
                    'rx="5" fill="#eff6ff" stroke="#93c5fd"/>'
                )
                elements.append(
                    f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
                    'font-family="Arial, sans-serif" font-size="12" '
                    f'font-weight="700" fill="#1d4ed8">tag sim {weight:.3f}</text>'
                )

    for name, (x, y) in sorted(positions.items(), key=lambda item: (group_of.get(item[0], ""), tiers.get(item[0], 0), item[0])):
        is_seed = name in seeds
        fill, text_color, radius = node_style(tiers.get(name, 99), is_seed)
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" '
            'stroke="#0f172a" stroke-width="1.2"/>'
        )
        lines = label_lines(name, 15 if not is_seed else 18)
        first_y = y - (len(lines) - 1) * 6 + 4
        for i, line in enumerate(lines):
            elements.append(
                f'<text x="{x:.1f}" y="{first_y + i * 12:.1f}" text-anchor="middle" '
                'font-family="Arial, sans-serif" '
                f'font-size="{11 if not is_seed else 12}" font-weight="700" '
                f'fill="{text_color}">{esc(line)}</text>'
            )

    legend_x = 40
    legend_y = 745
    legend_items = [
        ("seed", "#172554"),
        ("tier1 co-negative direct", "#f97316"),
        ("tier4 two-hop", "#7c3aed"),
    ]
    for index, (label, color) in enumerate(legend_items):
        x = legend_x + index * 230
        elements.append(f'<circle cx="{x}" cy="{legend_y}" r="10" fill="{color}"/>')
        elements.append(
            f'<text x="{x + 18}" y="{legend_y + 5}" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#334155">{esc(label)}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    output_path = output_dir / "seed_groups_harvested_JG87919_BtcKing1111.svg"
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 3:
        raise ValueError("This renderer expects exactly three seeds for the overview layout.")
    output_path = draw_svg(
        seeds=args.seeds,
        input_dir=args.input_dir,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
