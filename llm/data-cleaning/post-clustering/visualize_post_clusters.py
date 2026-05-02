from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import cluster_analyzed_posts as cp

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CLUSTERED_PATH = OUTPUT_DIR / "post_clusters.csv"
SUMMARY_PATH = OUTPUT_DIR / "cluster_summary.csv"
HTML_PATH = OUTPUT_DIR / "cluster_visualization.html"

MAX_POINTS = 8000
PLOT_NUMERIC_COLUMNS = [
    "sentiment_score",
    "manipulative_rhetoric_score",
    "high_manipulation_flag",
    "analysis_char_len",
    "word_count",
    "num_comments",
    "analyzed_comment_count",
    "avg_feedback_score",
    "supportive_comment_ratio",
    "oppositional_comment_ratio",
    "mixed_comment_ratio",
    "controversy_score",
    "tag_call_to_action",
    "tag_urgency",
    "tag_overconfidence",
    "tag_fear",
]
COLORS = [
    "#0f766e",
    "#dc2626",
    "#2563eb",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#059669",
    "#4f46e5",
    "#ea580c",
    "#0891b2",
]


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric_matrix = cp.zscore(df[PLOT_NUMERIC_COLUMNS].to_numpy(dtype=np.float32))
    text_matrix = cp.hashed_text_features(df["analysis_text"], dim=48)
    return np.concatenate([numeric_matrix, text_matrix], axis=1)


def choose_points(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()

    suspicious = df[df["cluster_rank_by_suspicion"] <= 3]
    remaining = df[df["cluster_rank_by_suspicion"] > 3]
    take_n = max(max_points - len(suspicious), 0)
    sample = remaining.sample(n=take_n, random_state=42, replace=False)
    return pd.concat([suspicious, sample], ignore_index=True)


def normalize(values: np.ndarray) -> list[float]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return [0.5] * len(values)
    return ((values - vmin) / (vmax - vmin)).round(6).tolist()


def make_payload(df: pd.DataFrame, summary: pd.DataFrame) -> dict:
    sampled = choose_points(df, MAX_POINTS).reset_index(drop=True)
    coords = pca_2d(build_feature_matrix(sampled))
    sampled["x"] = normalize(coords[:, 0])
    sampled["y"] = normalize(coords[:, 1])

    points = []
    for _, row in sampled.iterrows():
        title = str(row["title"])[:180]
        text_preview = str(row["analysis_text"])[:240].replace("\n", " ")
        points.append(
            {
                "x": row["x"],
                "y": row["y"],
                "cluster": int(row["cluster"]),
                "cluster_rank": int(row["cluster_rank_by_suspicion"]),
                "suspicion": float(row["cluster_suspicion_score"]),
                "manipulation": float(row["manipulative_rhetoric_score"]),
                "feedback": float(row["avg_feedback_score"]),
                "opposition": float(row["oppositional_comment_ratio"]),
                "title": title,
                "text": text_preview,
                "comments": float(row["num_comments"]),
                "length": float(row["analysis_char_len"]),
            }
        )

    clusters = []
    for _, row in summary.iterrows():
        cluster_id = int(row["cluster"])
        clusters.append(
            {
                "cluster": cluster_id,
                "color": COLORS[cluster_id % len(COLORS)],
                "size": int(row["size"]),
                "share": float(row["share"]),
                "suspicion": float(row["suspicion_score"]),
                "manipulation": float(row["avg_manipulative_rhetoric_score"]),
                "opposition": float(row["avg_oppositional_comment_ratio"]),
                "feedback": float(row["avg_feedback_score"]),
                "top_terms": row["top_terms"],
                "sample_titles": row["sample_titles"],
            }
        )

    return {"points": points, "clusters": clusters, "sampled_points": len(sampled), "total_points": len(df)}


def build_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Post Clustering Visualization</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255,255,255,0.88);
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #b45309;
      --line: rgba(31,41,55,0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180,83,9,0.16), transparent 32%),
        radial-gradient(circle at bottom right, rgba(15,118,110,0.15), transparent 28%),
        var(--bg);
    }}
    .wrap {{
      display: grid;
      grid-template-columns: minmax(320px, 1.15fr) minmax(280px, 420px);
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 12px 40px rgba(31,41,55,0.08);
      backdrop-filter: blur(10px);
    }}
    .chart-panel {{
      padding: 18px;
      display: flex;
      flex-direction: column;
    }}
    .side-panel {{
      padding: 18px;
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1.05;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 14px;
      font-size: 15px;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      font-size: 13px;
    }}
    #canvas-wrap {{
      position: relative;
      flex: 1;
      min-height: 560px;
      border-radius: 18px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.82)),
        repeating-linear-gradient(to right, transparent, transparent 79px, rgba(31,41,55,0.04) 80px),
        repeating-linear-gradient(to bottom, transparent, transparent 79px, rgba(31,41,55,0.04) 80px);
      border: 1px solid var(--line);
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      max-width: 320px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(17,24,39,0.92);
      color: white;
      font-size: 13px;
      line-height: 1.35;
      transform: translate(12px, 12px);
      opacity: 0;
      transition: opacity 120ms ease;
    }}
    .cluster-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      margin-bottom: 12px;
      background: rgba(255,255,255,0.72);
    }}
    .cluster-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 8px;
      vertical-align: middle;
    }}
    .cluster-title {{
      font-size: 17px;
      font-weight: 700;
    }}
    .score {{
      font-size: 13px;
      color: var(--accent);
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .terms {{
      font-size: 14px;
      line-height: 1.45;
      margin-top: 6px;
    }}
    .sample {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.4;
    }}
    @media (max-width: 980px) {{
      .wrap {{ grid-template-columns: 1fr; }}
      #canvas-wrap {{ min-height: 460px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel chart-panel">
      <h1>Post Clusters</h1>
      <div class="sub">PCA projection of final post features. Each point is one post, colored by cluster. Higher-ranked suspicious clusters are outlined more heavily.</div>
      <div class="meta">
        <div class="pill">Total cleaned posts: <strong id="total-points"></strong></div>
        <div class="pill">Points shown: <strong id="sampled-points"></strong></div>
        <div class="pill">Clusters: <strong id="cluster-count"></strong></div>
      </div>
      <div id="canvas-wrap">
        <canvas id="chart"></canvas>
        <div id="tooltip" class="tooltip"></div>
      </div>
    </section>
    <aside class="panel side-panel">
      <h1 style="font-size: 26px;">Suspicion View</h1>
      <div class="sub">Top clusters are sorted by a suspicion score using Gemini rhetoric labels and comment feedback patterns.</div>
      <div id="cluster-list"></div>
    </aside>
  </div>
  <script>
    const payload = {data_json};
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');
    const wrap = document.getElementById('canvas-wrap');
    const clusterMap = new Map(payload.clusters.map(c => [c.cluster, c]));
    const points = payload.points;
    const pointRadius = p => p.cluster_rank <= 3 ? 4.6 : 3.2;

    document.getElementById('total-points').textContent = payload.total_points.toLocaleString();
    document.getElementById('sampled-points').textContent = payload.sampled_points.toLocaleString();
    document.getElementById('cluster-count').textContent = payload.clusters.length;

    const clusterList = document.getElementById('cluster-list');
    payload.clusters.forEach((cluster, idx) => {{
      const card = document.createElement('div');
      card.className = 'cluster-card';
      card.innerHTML = `
        <div class="cluster-head">
          <div class="cluster-title"><span class="dot" style="background:${{cluster.color}}"></span>Cluster ${{cluster.cluster}}</div>
          <div class="score">Rank #${{idx + 1}} | Score ${{cluster.suspicion.toFixed(2)}}</div>
        </div>
        <div class="muted">Size ${{cluster.size.toLocaleString()}} | Share ${{(cluster.share * 100).toFixed(2)}}%</div>
        <div class="muted">Manipulation ${{cluster.manipulation.toFixed(2)}} | Opposition ${{(cluster.opposition * 100).toFixed(1)}}% | Feedback ${{cluster.feedback.toFixed(2)}}</div>
        <div class="terms"><strong>Top terms:</strong> ${{cluster.top_terms}}</div>
        <div class="sample"><strong>Sample titles:</strong> ${{cluster.sample_titles}}</div>
      `;
      clusterList.appendChild(card);
    }});

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = wrap.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function project(point) {{
      const rect = wrap.getBoundingClientRect();
      const pad = 26;
      return {{
        x: pad + point.x * (rect.width - pad * 2),
        y: pad + (1 - point.y) * (rect.height - pad * 2),
      }};
    }}

    function draw() {{
      const rect = wrap.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      for (const point of points) {{
        const pos = project(point);
        const cluster = clusterMap.get(point.cluster);
        const radius = pointRadius(point);
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = cluster.color + 'bb';
        ctx.fill();
        if (point.cluster_rank <= 3) {{
          ctx.lineWidth = 1.3;
          ctx.strokeStyle = '#111827';
          ctx.stroke();
        }}
      }}
    }}

    function hitTest(mx, my) {{
      let best = null;
      let bestDist = Infinity;
      for (const point of points) {{
        const pos = project(point);
        const radius = pointRadius(point) + 3;
        const dx = mx - pos.x;
        const dy = my - pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= radius && dist < bestDist) {{
          best = point;
          bestDist = dist;
        }}
      }}
      return best;
    }}

    wrap.addEventListener('mousemove', event => {{
      const rect = wrap.getBoundingClientRect();
      const hit = hitTest(event.clientX - rect.left, event.clientY - rect.top);
      if (!hit) {{
        tooltip.style.opacity = 0;
        return;
      }}
      const cluster = clusterMap.get(hit.cluster);
      tooltip.style.opacity = 1;
      tooltip.style.left = `${{event.clientX - rect.left}}px`;
      tooltip.style.top = `${{event.clientY - rect.top}}px`;
      tooltip.innerHTML = `
        <strong>Cluster ${{hit.cluster}}</strong> | suspicion ${{hit.suspicion.toFixed(2)}}<br>
        <strong>Title:</strong> ${{hit.title}}<br>
        <strong>Preview:</strong> ${{hit.text}}<br>
        <strong>Manipulation:</strong> ${{hit.manipulation.toFixed(0)}} | <strong>Feedback:</strong> ${{hit.feedback.toFixed(0)}} | <strong>Opposition:</strong> ${{(hit.opposition * 100).toFixed(1)}}%<br>
        <strong>Comments:</strong> ${{hit.comments}} | <strong>Chars:</strong> ${{hit.length}}<br>
        <strong>Top terms:</strong> ${{cluster.top_terms}}
      `;
    }});

    wrap.addEventListener('mouseleave', () => {{
      tooltip.style.opacity = 0;
    }});

    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""


def main() -> None:
    df = pd.read_csv(CLUSTERED_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    payload = make_payload(df, summary)
    HTML_PATH.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
