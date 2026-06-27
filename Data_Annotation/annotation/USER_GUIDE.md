# OCR Annotation Review Guide

This guide describes how reviewers prepare data, launch the interface, edit records, and recover from common issues.

## 1. Prepare the Directory

Install dependencies:

```bash
cd Data_Annotation/annotation
pip install -r requirements.txt
```

Place images in `images/`, for example:

```text
annotation/
├── app.py
├── images/
│   ├── q001.jpg
│   ├── q002.jpg
│   └── q003.png
└── results.json
```

If `results.json` does not exist yet, the UI will create it on the first save.

## 2. Launch the UI

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## 3. Interface Overview

- Top bar: previous/next navigation, direct jump, progress display
- Left panel: image viewer with zoom, drag, and reset
- Right panel: editable form fields generated from `SCHEMA_CONFIG`
- Bottom panel: diff preview and save status

## 4. Editing Workflow

1. Open an image.
2. Inspect the handwriting in the viewer.
3. Update `idx`, `QuestionType`, `transcription`, and `final_answer` as needed.
4. Wait for auto-save or press `Ctrl+S` / `Cmd+S` for an immediate save.

The UI writes all records into the shared `results.json` file and updates an existing record when the `filename` key matches.

## 5. Navigation Rules

You can switch images by:

- clicking the previous or next button
- entering a target index and pressing the jump button
- using the left and right arrow keys when focus is not inside an input field

If there are unsaved edits, the UI shows a confirmation panel with:

- `Save and switch`
- `Discard and switch`
- `Cancel`

## 6. Common Issues

### No images appear

- confirm that `images/` exists
- confirm that it contains `jpg`, `jpeg`, or `png` files

### Saving is disabled

Typical causes:

- `results.json` is invalid JSON
- the top-level value is not an array
- duplicate `filename` values exist

### Image preview fails

The UI shows a placeholder image with the error message. Check whether the source file is readable by standard image tools.

### Old records do not populate automatically

The current UI requires `filename` to match the real file name inside `images/`.
