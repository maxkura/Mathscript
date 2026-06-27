# MathScript

MathScript 是一个面向高中数学手写作答图像的 OCR benchmark，用于评测模型对手写文本、数学公式、结构化转写和阅读顺序的还原能力。

本仓库按完整发行包组织：正式评测数据、标准答案、公开模型预测结果、评测脚本和辅助文档均使用仓库相对路径。除非特别说明，下面所有命令都假设当前工作目录是仓库根目录。

## 数据集构成

正式评测只使用当前官方 `GT` 文件和对应图片，不使用历史或中间数据文件。

| 内容 | 路径 | 说明 |
| --- | --- | --- |
| 手写作答图片 | `Data_Annotation/annotation/images/` | 正式评测图像目录 |
| 标准答案 `GT` | `data/GT/extracted_gt.json` | 评测使用的标准答案数据流 |
| 模型预测 `predict` | `output/predict/<model_name>/predict.json` | 可直接进入评测的模型预测文件 |
| 评测输出 | `output/metric/` | `eval` 生成的样本级、模型级和排行榜结果 |

正式集统计如下：

| 指标 | 数量 |
| --- | ---: |
| 图片数量 | 446 |
| `GT` 记录数 | 446 |
| `ConstructedResponse` | 446 |
| `MultipleChoice` | 0 |
| `FillBlank` | 0 |
| 结构化样本 | 403 |
| `[UNK]` passthrough 样本 | 43 |
| 结构化片段 | 6086 |
| 含公式样本 | 402 |
| 公式槽位 | 6702 |

数据集不划分 train/dev/test，正式发行版提供一个统一评测集。

公开的模型预测目录包括：

- `claude-sonnet-4-5-20250929`
- `doubao-1-5-vision-pro-32K`
- `doubao-seed-1.6-vision`
- `gemini-2.5-flash`
- `gpt-4o`
- `grok-4-0709`
- `kimi-k2.5`
- `qwen3-vl-plus`
- `qwen3.5-flash`
- `qwen3.5-plus`

## 数据格式

`GT` 和正式 `predict` 都是 JSON 数组。每条记录至少包含以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `filename` | string | 图片文件名，也是样本配对主键 |
| `idx` | integer 或 null | 题目序号，评测中作为保留字段 |
| `QuestionType` | string | 当前正式集均为 `ConstructedResponse` |
| `transcription` | array 或 string | 结构化转写；`[UNK]` 样本可为 passthrough 字符串 |
| `formula_list` | array | 从转写中抽取的公式槽位，可为空数组 |
| `final_answer` | string | 最终答案文本 |

结构化 `transcription` 的元素至少包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `seq` | integer | 样本内片段顺序，从 1 开始 |
| `question_id` | string | 子题编号，如 `Q001` |
| `content` | string | 当前片段内容 |

正式 `predict` 中的结构化 `transcription` 还可以包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `gt_seq` | integer 或 null | 与 `GT` 对齐后的目标片段序号 |
| `match_status` | string | 对齐状态，如 `matched` 或 `unmatched` |
| `match_score` | number | 对齐相似度分数 |

`formula_list` 的元素至少包含 `formula_seq`、`seq`、`question_id`、`formula`。在正式 `predict` 中，公式项还可以包含 `gt_seq`，表示该公式对齐到的 `GT` 片段。

最小 `predict.json` 示例：

```json
[
  {
    "filename": "row_C_1.jpg",
    "idx": 1,
    "QuestionType": "ConstructedResponse",
    "transcription": [
      {
        "seq": 1,
        "question_id": "Q001",
        "content": "解得 x+1=2",
        "gt_seq": 1,
        "match_status": "matched",
        "match_score": 0.98
      }
    ],
    "formula_list": [
      {
        "formula_seq": 1,
        "seq": 1,
        "question_id": "Q001",
        "gt_seq": 1,
        "formula": "x+1=2"
      }
    ],
    "final_answer": "x=1"
  }
]
```

## 程序说明

| 程序 | 作用 |
| --- | --- |
| `Data_Annotation/annotation/app.py` | 启动人工标注与复核界面，读取图片并维护标注结果 |
| `Data_Annotation/annotation/preannotate_kimi_k2_5.py` | 调用视觉模型生成预标注结果，作为人工校正起点 |
| `Data_Annotation/annotation/extract_gt.py` | 从人工校正后的标注结果提取正式 `GT` |
| `Data_Annotation/annotation/extraction_utils.py` | 提供 `GT` 提取所需的结构化、公式抽取和错误报告逻辑 |
| `Data_Annotation/annotation/remove_transcription_spaces.py` | 清理 `transcription` 中的空格噪声 |
| `eval/predict.py` | 调用待评测视觉模型，生成原始 `results.json` |
| `eval/reduce_format_noise.py` | 对原始预测中的 `transcription` 做格式噪声削弱 |
| `eval/predict_convert.py` | 将 `results.json` 或 `result_less_format_noise.json` 转换为正式 `predict.json` |
| `eval/eval.py` | 读取 `GT` 与正式 `predict`，执行配对、指标计算和排行榜汇总 |
| `eval/metric.py` | 实现 stem、formula、reading order、refusal 和 composite score 等指标 |
| `output/metric/export_leaderboard_csv.py` | 将 `leaderboard_summary.json` 导出为 CSV |
| `eval/build_threshold_report.py` | 可选：汇总多阈值评测结果并生成 Markdown/CSV 表格 |
| `eval/protocol_debias.py` | 可选：构建协议去偏版本的 `predict` 并运行官方评测流程 |

## 评测流程

主流程从已有正式 `predict.json` 开始。将待评测模型的预测文件放在：

```text
output/predict/<model_name>/predict.json
```

评测所有包含 `predict.json` 的模型：

```bash
python3 eval/eval.py \
  --gt data/GT/extracted_gt.json \
  --predict-root output/predict \
  --output-root output/metric
```

只评测单个模型：

```bash
python3 eval/eval.py \
  --model gpt-4o \
  --gt data/GT/extracted_gt.json \
  --predict-root output/predict \
  --output-root output/metric
```

主要输出包括：

| 输出 | 说明 |
| --- | --- |
| `output/metric/leaderboard_summary.json` | 所有成功评测模型的排行榜汇总 |
| `output/metric/<model_name>/sample_metrics.json` | 单模型样本级评测结果 |
| `output/metric/<model_name>/overall_metrics.json` | 单模型整体指标 |
| `output/metric/<model_name>/eval_errors.json` | 缺失预测、额外预测或 fatal error 报告 |

导出排行榜 CSV：

```bash
python3 output/metric/export_leaderboard_csv.py \
  -i output/metric/leaderboard_summary.json \
  -o output/metric/leaderboard_summary.csv
```

## API Key 配置

从已有 `predict.json` 开始运行正式评测不需要任何 API key。只有在重新生成预测、执行格式噪声削弱、预标注或其他会调用模型 API 的流程时，评测者才需要自行配置密钥。

不要把真实 API key 写入仓库、README、脚本或提交记录。推荐在本地 shell、CI secret 或密钥管理系统中设置环境变量；也可以在支持的脚本中临时使用 `--api-key` 和 `--base-url` 参数覆盖。

常用环境变量如下：

| 环境变量 | 用途 |
| --- | --- |
| `OPENAI_COMPAT_API_KEY` | `gpt-4o`、`grok-4-0709`、`claude-sonnet-4-5-20250929`、`gemini-2.5-flash` 等 OpenAI-compatible 入口 |
| `KIMI_API_KEY` | `kimi-k2.5` 预测与 `eval/reduce_format_noise.py` 默认清洗模型 |
| `IFLOW_API_KEY` | `qwen3-vl-plus` |
| `DASHSCOPE_API_KEY` | `qwen3.5-flash`、`qwen3.5-plus` 以及题型分类脚本 |
| `DOUBAO_API_KEY` | `doubao-1-5-vision-pro-32K`、`doubao-seed-1.6-vision` |
| `MIMO_API_KEY` | `mimo-v2-omni` |
| `ANNOTATION_API_KEY` | `Data_Annotation/annotation/preannotate_kimi_k2_5.py` 预标注 |
| `ANNOTATION_BASE_URL` | 预标注使用的 OpenAI-compatible endpoint |
| `ANNOTATION_MODEL` | 预标注使用的模型名 |

示例配置如下，请把尖括号内容替换为自己的值：

```bash
export OPENAI_COMPAT_API_KEY="<YOUR_OPENAI_COMPAT_API_KEY>"
export KIMI_API_KEY="<YOUR_KIMI_API_KEY>"
export IFLOW_API_KEY="<YOUR_IFLOW_API_KEY>"
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"
export DOUBAO_API_KEY="<YOUR_DOUBAO_API_KEY>"
export MIMO_API_KEY="<YOUR_MIMO_API_KEY>"
export ANNOTATION_API_KEY="<YOUR_ANNOTATION_API_KEY>"
export ANNOTATION_BASE_URL="<YOUR_OPENAI_COMPATIBLE_ENDPOINT>"
export ANNOTATION_MODEL="<YOUR_ANNOTATION_MODEL>"
```

## 可选：从图片生成 predict

如果需要从原始图片重新生成某个模型的正式 `predict.json`，可以按以下顺序运行：

```bash
python3 eval/predict.py --model gpt-4o
python3 eval/reduce_format_noise.py --model gpt-4o
python3 eval/predict_convert.py --model gpt-4o --gt data/GT/extracted_gt.json
python3 eval/eval.py --model gpt-4o --gt data/GT/extracted_gt.json --predict-root output/predict --output-root output/metric
```

`predict.py` 会调用模型 API，运行前需要按目标模型配置相应的 API key 或通过命令行参数传入 `--api-key`、`--base-url`。具体参数可查看：

```bash
python3 eval/predict.py --help
python3 eval/reduce_format_noise.py --help
python3 eval/predict_convert.py --help
```

## 快速检查

仅检查命令行入口是否与文档一致：

```bash
python3 eval/eval.py --help
python3 eval/predict_convert.py --help
python3 output/metric/export_leaderboard_csv.py --help
```

完整评测会读取正式数据和模型预测文件，发行包补齐 `data/`、`Data_Annotation/annotation/images/` 与 `output/predict/` 后即可运行。
