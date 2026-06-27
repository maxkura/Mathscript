# Prediction Pipeline Usage

This document covers the raw prediction stage (`predict.py`), the optional cleanup stage (`reduce_format_noise.py`), and the conversion stage (`predict_convert.py`).

## 1. Raw Prediction Generation

Run:

```bash
python3 eval/predict.py --model gpt-4o
```

Purpose:

1. Read input images.
2. Read the OCR prompt template.
3. Call a vision model API.
4. Store raw model output.
5. Parse JSON when possible.
6. Write flattened records to `results.json`.

Default paths:

- images: `Data_Annotation/annotation/images`
- prompt: `prompt/preannotation_prompt.txt`
- output root: `output/predict`

Default per-model output:

```text
output/predict/<model_name>/results.json
```

Supported model names:

- `gpt-4o`
- `grok-4-0709`
- `claude-sonnet-4-5-20250929`
- `gemini-2.5-flash`
- `kimi-k2.5`
- `qwen3-vl-plus`
- `qwen3.5-flash`
- `qwen3.5-plus`
- `mimo-v2-omni`
- `doubao-1-5-vision-pro-32K`
- `doubao-seed-1.6-vision`

Useful options:

```bash
python3 eval/predict.py --model kimi-k2.5
python3 eval/predict.py --model gpt-4o --output output/predict/gpt-4o/results.json
python3 eval/predict.py --model gpt-4o --max-retries 8 --max-tokens 6000
python3 eval/predict.py --model gpt-4o --overwrite
```

The raw prediction stage does not build final evaluation-ready `predict.json`.

## 2. Optional Transcription Cleanup

Run:

```bash
python3 eval/reduce_format_noise.py
```

Purpose:

- process one record at a time
- only edit the `transcription` field
- preserve other fields
- write `result_less_format_noise.json` under each model directory

Default prompt file:

```text
prompt/predict_result_cleanup_prompt.txt
```

Useful options:

```bash
python3 eval/reduce_format_noise.py --model gpt-4o
python3 eval/reduce_format_noise.py --overwrite
python3 eval/reduce_format_noise.py --prompt-file prompt/predict_result_cleanup_prompt.txt
```

## 3. Convert Raw Results into Final `predict.json`

Run:

```bash
python3 eval/predict_convert.py --model gpt-4o
```

Purpose:

1. Read `result_less_format_noise.json` when available, otherwise fall back to `results.json`.
2. Read `data/GT/extracted_gt.json`.
3. Convert successful raw records into the normalized `predict.json` format.
4. Write `predict_conversion_errors.json` alongside the converted output.

Default outputs:

- `output/predict/<model_name>/predict.json`
- `output/predict/<model_name>/predict_conversion_errors.json`

Useful options:

```bash
python3 eval/predict_convert.py --model gpt-4o
python3 eval/predict_convert.py --model gpt-4o --match-threshold 0.50
python3 eval/predict_convert.py --batch --output-root output/predict
```

## 4. Recommended End-to-End Order

```bash
python3 eval/predict.py --model gpt-4o
python3 eval/reduce_format_noise.py --model gpt-4o
python3 eval/predict_convert.py --model gpt-4o
python3 eval/eval.py --model gpt-4o
```

## 5. Output Notes

`results.json` stores raw prediction records such as:

- `ImgReal`
- `idx`
- `QuestionType`
- `transcription`
- `final_answer`
- `raw_output`
- `predict_status`
- `error_message`

`predict.json` stores the normalized evaluation-facing structure.

`predict_conversion_errors.json` is an output report only. It is not used as an input file.
