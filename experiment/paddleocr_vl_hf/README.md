# PaddleOCR-VL HF Smoke Test

This directory contains a minimal smoke test for the PaddleOCR-VL Hugging Face setup used during experiments.

## Environment

Activate the environment you prepared for this experiment, for example:

```bash
conda activate paddleocr_vl_hf
```

## Run the Default Smoke Test

```bash
python experiment/paddleocr_vl_hf/run_ocr_smoke_test.py
```

The default command assumes you have a local benchmark image available at the repository-relative path expected by the script. If your submission bundle omits test images, pass `--image` explicitly.

## Run with Explicit Paths

```bash
python experiment/paddleocr_vl_hf/run_ocr_smoke_test.py \
  --image data/test/clean/row_C_6.jpg \
  --output-json experiment/paddleocr_vl_hf/output/row_C_6.result.json
```

## Notes

- All example paths are repository-relative.
- The `output/` directory is created on demand and is intentionally omitted from the submission bundle.
- The script is intended for environment validation rather than benchmark scoring.
