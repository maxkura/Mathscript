# Analysis Figures

This directory stores the figure renderers used for paper-facing diagrams.

## GT Data Construction Figure

Render with:

```bash
python3 analysis/render_gt_pipeline.py
```

Outputs:

- `figures/gt_data_construction_pipeline_en.png`
- `figures/gt_data_construction_pipeline_en.svg`

## Evaluation Pipeline Figure

Render with:

```bash
python3 analysis/render_eval_pipeline.py
```

Outputs:

- `figures/eval_pipeline_en.png`
- `figures/eval_pipeline_en.svg`

## Notes

- The renderers can use system-installed Noto CJK fonts when needed. No local font package is bundled in this submission copy.
- The SVG files are lightweight wrappers around the rendered PNG images so they remain easy to embed in slides or papers.
