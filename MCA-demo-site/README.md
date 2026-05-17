# MCA Demo Site

Static dashboard for presenting MCA pipeline results.

## What It Shows

- Suspicious coordination groups from seed expansion and validation.
- Group relationship graphs using co-negative and temporal evidence.
- Stage 2 evidence: temporal synchrony and activation window overlap.
- Individual abnormal accounts as a separate output stream.
- Client/Demo audience modes.
- Chinese/English language switching.

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
