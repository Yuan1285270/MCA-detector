# MCA Demo Site

Static multi-page dashboard for presenting MCA pipeline results as a client-facing risk review product and a research demo.

This folder is separate from the core analysis pipeline. It is a presentation layer only:

```text
coordination-expansion/output -> MCA-demo-site/data/demo-data.js -> browser dashboard
```

The demo site does not compute MCA scores, discover groups, verify temporal synchrony, or decide final review priority. Those outputs are produced by the project pipeline before the site reads them.

## What It Shows

- `index.html`: executive overview and review priorities.
- `groups.html`: risk group queue, relationship graph, and pair evidence.
- `accounts.html`: individual abnormal account stream.
- `methodology.html`: pipeline and signal-pruning summary.
- Client/Demo audience modes and Chinese/English language switching.
- MCA Sentinel SVG logo and enterprise-style navigation.

## Build Data

Run from the repository root:

```bash
python3 MCA-demo-site/scripts/build_demo_data.py
```

This writes:

```text
MCA-demo-site/data/demo-data.js
```

The generated bundle is built from `coordination-expansion/output`.

## Run Locally

```bash
cd MCA-demo-site
python3 -m http.server 5177
```

Then open:

```text
http://localhost:5177
```

## Design Note

The demo intentionally separates:

1. Group-level suspicious coordination
2. Account-level abnormal manipulation

This keeps the site from overclaiming that every abnormal account is part of a coordinated group.

The default view is **Client mode** in Chinese. Client mode frames the output as a risk review queue. Demo mode keeps the research and pipeline language visible for presentation or debugging.

The site should not present MCA score as a guilt score. The intended story is:

```text
MCA ranking finds suspicious entry points.
Graph expansion finds candidate groups around those entries.
Temporal synchrony decides which candidate groups deserve higher review priority.
```
