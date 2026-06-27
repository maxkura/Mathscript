# `eval.py` Usage

This document summarizes the common ways to run `eval.py` and `test_eval.py`.

All commands below assume the current working directory is the repository root.

## Evaluate All Models with Default Paths

```bash
python3 eval/eval.py
```

Default inputs:

- `data/GT/extracted_gt.json`
- `output/predict/<model_name>/predict.json`

Default outputs:

- `output/metric/leaderboard_summary.json`
- `output/metric/<model_name>/sample_metrics.json`
- `output/metric/<model_name>/overall_metrics.json`
- `output/metric/<model_name>/eval_errors.json`

## Evaluate a Single Model

```bash
python3 eval/eval.py --model qwen3-vl-plus
```

## Override Paths

```bash
python3 eval/eval.py \
  --gt data/GT/extracted_gt.json \
  --predict-root output/predict \
  --output-root output/metric
```

## Adjust Thresholds or Weights

```bash
python3 eval/eval.py \
  --match-threshold 0.50 \
  --w-slt 0.4 \
  --w-opt 0.6 \
  --alpha-stem 0.40 \
  --alpha-formula 0.35 \
  --alpha-ros 0.20 \
  --alpha-refusal 0.05
```

## Run the Tests

Recommended:

```bash
python3 -m unittest eval/test_eval.py
```

Also supported:

```bash
python3 eval/test_eval.py
```

## Notes

- `--match-threshold` affects line alignment in conversion and evaluation.
- `--w-slt` and `--w-opt` must sum to `1`.
- `--alpha-*` weights must sum to `1`.
- Some tests intentionally trigger fatal edge-case logging on stderr. The run is still successful as long as `unittest` reports `OK`.
