# OCR Annotation Review System

This directory contains the Gradio-based review tool used to inspect and correct OCR annotations.

The application reads images from `images/` and stores all records in the colocated `results.json`. Every path is derived from the location of `app.py`, so the directory can be moved without editing absolute paths.

The submission bundle keeps the annotation code and docs, but omits local `images/` and generated JSON artifacts so the supplementary material stays lightweight and double-blind safe.

## Layout

```text
annotation/
├── app.py
├── extract_gt.py
├── extraction_utils.py
├── preannotate_kimi_k2_5.py
├── remove_transcription_spaces.py
├── requirements.txt
├── README.md
├── USER_GUIDE.md
├── images/
└── results.json
```

## Main Components

- `app.py`: interactive review UI
- `preannotate_kimi_k2_5.py`: batch pre-annotation script for image-level OCR records
- `extract_gt.py`: converts reviewed records into benchmark GT format
- `extraction_utils.py`: shared helpers for GT extraction
- `results.json`: shared annotation store keyed by `filename`

## Setup

Use Python 3.10 or newer.

```bash
cd Data_Annotation/annotation
pip install -r requirements.txt
```

## Pre-Annotation

Run:

```bash
python preannotate_kimi_k2_5.py
```

Default behavior:

- scans `images/`
- writes to `results.json`
- keeps one record per image
- skips existing successful records unless `--overwrite` is passed

Environment variables:

```bash
export ANNOTATION_API_KEY="<YOUR_API_KEY>"
export ANNOTATION_BASE_URL="<YOUR_OPENAI_COMPATIBLE_ENDPOINT>"
export ANNOTATION_MODEL="<YOUR_MODEL_NAME>"
```

`ANNOTATION_API_KEY` and `ANNOTATION_BASE_URL` are required unless you pass overrides explicitly.

## GT Extraction

Run:

```bash
python extract_gt.py
```

Default behavior:

- reads `results.json`
- writes `extracted_gt.json`
- writes `extraction_errors.json`
- keeps `MultipleChoice` and `FillBlank` records as passthrough values
- further structures `ConstructedResponse` records when sub-question markers are present

## Launch the UI

Run:

```bash
python app.py
```

The default local address is:

```text
http://127.0.0.1:7860
```

## Data Notes

- `results.json` must be a top-level JSON array.
- Each record must use `filename` as the primary image key.
- Extra fields outside the core schema are preserved during save operations.

See [USER_GUIDE.md](./USER_GUIDE.md) for step-by-step reviewer instructions.
