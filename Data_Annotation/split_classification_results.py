import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


QUESTION_TYPES = [
    "MultipleChoice",
    "FillBlank",
    "ConstructedResponse",
    "Unclassified",
]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "classification_results.json"
DEFAULT_SPLIT_JSON_DIR = SCRIPT_DIR / "split_json"
DEFAULT_SPLIT_IMAGE_DIR = SCRIPT_DIR / "split_images"


def load_results(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    return [item for item in data if isinstance(item, dict)]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def normalize_question_type(value: Any) -> str:
    if isinstance(value, str) and value in QUESTION_TYPES[:-1]:
        return value
    return "Unclassified"


def ensure_unique_path(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 1
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_image(record: Dict[str, Any], target_dir: Path) -> None:
    img_real = record.get("ImgReal")
    image_name = record.get("ImageName")

    if not isinstance(img_real, str) or not img_real:
        return
    if not isinstance(image_name, str) or not image_name:
        image_name = Path(img_real).name

    source_path = Path(img_real)
    if not source_path.exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = ensure_unique_path(target_dir, image_name)
    shutil.copy2(source_path, target_path)


def split_results(
    records: List[Dict[str, Any]],
    split_json_dir: Path,
    split_image_dir: Path,
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in QUESTION_TYPES}

    for record in records:
        question_type = normalize_question_type(record.get("QuestionType"))
        normalized_record = {
            "ImageName": record.get("ImageName", ""),
            "ImgReal": record.get("ImgReal", ""),
            "QuestionType": question_type,
            "status": record.get("status", ""),
            "error": record.get("error", ""),
        }
        grouped[question_type].append(normalized_record)
        copy_image(normalized_record, split_image_dir / question_type)

    for question_type, items in grouped.items():
        atomic_write_json(split_json_dir / f"{question_type}.json", items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split classification_results.json into per-type JSON files and image folders.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to classification_results.json.",
    )
    parser.add_argument(
        "--json-output-dir",
        type=str,
        default=str(DEFAULT_SPLIT_JSON_DIR),
        help="Directory for per-type JSON outputs.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        default=str(DEFAULT_SPLIT_IMAGE_DIR),
        help="Directory for per-type copied images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    split_json_dir = Path(args.json_output_dir).resolve()
    split_image_dir = Path(args.image_output_dir).resolve()

    records = load_results(input_path)
    split_results(records, split_json_dir, split_image_dir)

    print(f"Split JSON written to: {split_json_dir}")
    print(f"Split images written to: {split_image_dir}")


if __name__ == "__main__":
    main()
