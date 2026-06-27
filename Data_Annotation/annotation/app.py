# -*- coding: utf-8 -*-

import base64
import copy
import html
import json
import time
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
from PIL import Image, ImageDraw


# ==================== UI schema ====================
# Update this dictionary when the JSON schema changes. The form is generated
# directly from this config to keep the UI and stored fields in sync.
SCHEMA_CONFIG = {
    "filename": {
        "type": "text",
        "label": "Filename",
        "editable": False,
        "description": "Original image filename. Auto-generated and read-only.",
    },
    "idx": {
        "type": "number",
        "label": "Question Index",
        "editable": True,
        "description": "Question index in the benchmark.",
    },
    "QuestionType": {
        "type": "dropdown",
        "label": "Question Type",
        "editable": True,
        "choices": ["MultipleChoice", "FillBlank", "ConstructedResponse"],
        "description": "Select the question type.",
    },
    "sub_question_count": {
        "type": "number",
        "label": "Sub-question Count",
        "editable": True,
        "description": "Number of sub-questions in this sample.",
    },
    "transcription": {
        "type": "textarea",
        "label": "Transcription",
        "editable": True,
        "description": "Full OCR transcription for the sample.",
        "lines": 5,
    },
    "final_answer": {
        "type": "text",
        "label": "Final Answer",
        "editable": True,
        "description": "Reference answer. Use options for multiple choice and the answer text for fill-blank samples.",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FIELD_ORDER = list(SCHEMA_CONFIG.keys())
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
RESULTS_FILE = BASE_DIR / "results.json"


CUSTOM_CSS = """
body {
  background: #f4f5f8;
}

.app-shell {
  max-width: 1600px;
  margin: 0 auto;
}

.viewer-host,
.diff-host,
.status-host,
.confirm-host {
  border: 1px solid #d7dce5;
  border-radius: 14px;
  background: #ffffff;
}

.viewer-host {
  overflow: hidden;
}

.viewer-shell {
  display: flex;
  flex-direction: column;
  min-height: 620px;
  background:
    radial-gradient(circle at top left, rgba(45, 96, 196, 0.07), transparent 32%),
    linear-gradient(180deg, #fbfcff 0%, #f4f7fb 100%);
}

.viewer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #e6ebf2;
  color: #455066;
  font-size: 13px;
}

.viewer-stage {
  position: relative;
  flex: 1;
  min-height: 560px;
  cursor: grab;
  touch-action: none;
}

.viewer-stage.dragging {
  cursor: grabbing;
}

.viewer-canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.viewer-source {
  display: none;
}

.viewer-empty {
  display: grid;
  place-items: center;
  min-height: 560px;
  padding: 24px;
  color: #5a6476;
  font-size: 15px;
}

.viewer-note {
  font-size: 12px;
  color: #72809a;
}

.diff-host {
  padding: 14px 16px;
}

.diff-block {
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.diff-field {
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px dashed #d9deea;
}

.diff-field:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.diff-title {
  color: #25324b;
  font-weight: 700;
}

.diff-line {
  display: block;
  margin-top: 4px;
  padding: 2px 8px;
  border-radius: 8px;
}

.diff-line.del {
  color: #a23131;
  background: #fff1f1;
}

.diff-line.add {
  color: #17643e;
  background: #ecfff4;
}

.diff-line.same {
  color: #58657b;
  background: #f6f8fb;
}

.diff-clean {
  color: #5c6a80;
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 13px;
}

.status-host {
  padding: 12px 14px;
}

.status-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #2e394d;
}

.status-main {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  flex: 0 0 auto;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.04);
}

.status-label {
  font-size: 14px;
  font-weight: 600;
}

.status-time {
  font-size: 12px;
  color: #65758f;
}

.status-green {
  background: #21a168;
}

.status-yellow {
  background: #f1b53d;
}

.status-red {
  background: #db4d4d;
}

.confirm-host {
  padding: 14px 16px;
  border-color: #d7b86e;
  background: #fffbef;
}

.confirm-text {
  color: #52411b;
  font-size: 14px;
  margin-bottom: 10px;
}

.read-only-field input,
.read-only-field textarea {
  background: #f1f3f7 !important;
}

.hidden-action {
  display: none !important;
}
"""


APP_HEAD = """
<script>
(() => {
  if (window.__ocrAnnotationAppBootstrapped) {
    return;
  }
  window.__ocrAnnotationAppBootstrapped = true;

  const runtime = {
    autoSaveTimer: null,
  };

  const clickButton = (id) => {
    const root = document.getElementById(id);
    const button = root?.querySelector("button") || root;
    if (button) {
      button.click();
    }
  };

  const isTypingTarget = (target) => {
    if (!target) return false;
    const tag = (target.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
  };

  const isEditableFieldTarget = (target) => {
    const fieldRoot = target?.closest?.("[id^='field-']");
    if (!fieldRoot) {
      return false;
    }
    const control = fieldRoot.querySelector("input, textarea, select");
    if (!control) {
      return false;
    }
    return !control.disabled && !control.readOnly;
  };

  const clearAutoSave = () => {
    if (runtime.autoSaveTimer !== null) {
      window.clearTimeout(runtime.autoSaveTimer);
      runtime.autoSaveTimer = null;
    }
  };

  // Debounce auto-save in the browser so zooming and typing do not trigger
  // competing backend/UI updates.
  const scheduleAutoSave = () => {
    clearAutoSave();
    runtime.autoSaveTimer = window.setTimeout(() => {
      runtime.autoSaveTimer = null;
      clickButton("hidden-auto-save-button");
    }, 800);
  };

  document.addEventListener("input", (event) => {
    if (isEditableFieldTarget(event.target)) {
      scheduleAutoSave();
    }
  }, true);

  document.addEventListener("change", (event) => {
    if (isEditableFieldTarget(event.target)) {
      scheduleAutoSave();
    }
  }, true);

  document.addEventListener("click", (event) => {
    const actionRoot = event.target?.closest?.(
      "#nav-prev-button, #nav-next-button, #jump-button, #reset-view-button, " +
      "#confirm-save-button, #confirm-discard-button, #confirm-cancel-button, " +
      "#hidden-manual-save-button"
    );
    if (actionRoot) {
      clearAutoSave();
    }
  }, true);

  document.addEventListener("keydown", (event) => {
    const key = event.key;
    if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "s") {
      event.preventDefault();
      clearAutoSave();
      clickButton("hidden-manual-save-button");
      return;
    }

    if (isTypingTarget(event.target)) {
      return;
    }

    if (key === "ArrowLeft") {
      event.preventDefault();
      clearAutoSave();
      clickButton("nav-prev-button");
    } else if (key === "ArrowRight") {
      event.preventDefault();
      clearAutoSave();
      clickButton("nav-next-button");
    }
  }, { passive: false });

  const initViewer = (root) => {
    if (!root || root.dataset.viewerReady === "1") {
      return;
    }
    root.dataset.viewerReady = "1";

    const stage = root.querySelector(".viewer-stage");
    const canvas = root.querySelector(".viewer-canvas");
    const source = root.querySelector(".viewer-source");
    if (!stage || !canvas || !source) {
      return;
    }

    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.src = source.src;

    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const minScale = 0.1;
    const maxScale = 5.0;

    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

    const paintBackground = () => {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#eef2f7";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(24, 24, Math.max(0, canvas.width - 48), Math.max(0, canvas.height - 48));
    };

    const render = () => {
      paintBackground();
      if (!img.complete || !img.naturalWidth || !img.naturalHeight) {
        return;
      }
      ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, 0, 0);
    };

    const fitImage = () => {
      if (!img.naturalWidth || !img.naturalHeight || !canvas.width || !canvas.height) {
        return;
      }
      const fitScale = Math.min(
        canvas.width / img.naturalWidth,
        canvas.height / img.naturalHeight
      );
      scale = clamp(fitScale, minScale, maxScale);
      offsetX = (canvas.width - img.naturalWidth * scale) / 2;
      offsetY = (canvas.height - img.naturalHeight * scale) / 2;
      render();
    };

    const resizeCanvas = () => {
      const rect = stage.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width));
      canvas.height = Math.max(560, Math.floor(rect.height || 560));
      fitImage();
    };

    // Keep viewer interactions in the browser so pan/zoom stays responsive.
    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      if (!img.complete || !img.naturalWidth || !img.naturalHeight) {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      const worldX = (mouseX - offsetX) / scale;
      const worldY = (mouseY - offsetY) / scale;
      const delta = event.deltaY < 0 ? 1.1 : 0.9;
      const nextScale = clamp(scale * delta, minScale, maxScale);
      offsetX = mouseX - worldX * nextScale;
      offsetY = mouseY - worldY * nextScale;
      scale = nextScale;
      render();
    }, { passive: false });

    stage.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      stage.classList.add("dragging");
      stage.setPointerCapture(event.pointerId);
    });

    stage.addEventListener("pointerup", (event) => {
      dragging = false;
      stage.classList.remove("dragging");
      if (stage.hasPointerCapture(event.pointerId)) {
        stage.releasePointerCapture(event.pointerId);
      }
    });

    stage.addEventListener("pointercancel", () => {
      dragging = false;
      stage.classList.remove("dragging");
    });

    stage.addEventListener("pointermove", (event) => {
      if (!dragging) {
        return;
      }
      offsetX += event.clientX - lastX;
      offsetY += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      render();
    });

    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas();
    });
    resizeObserver.observe(stage);

    img.onload = () => {
      resizeCanvas();
    };

    img.onerror = () => {
      paintBackground();
      ctx.fillStyle = "#7a8698";
      ctx.font = "16px sans-serif";
      ctx.fillText("Failed to load image", 40, 60);
    };

    resizeCanvas();
  };

  const scanViewers = () => {
    document.querySelectorAll("[data-ocr-viewer='1']").forEach(initViewer);
  };

  const startObservers = () => {
    if (!document.body) {
      return;
    }
    const mutationObserver = new MutationObserver(() => {
      scanViewers();
    });
    mutationObserver.observe(document.body, { childList: true, subtree: true });
    scanViewers();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startObservers, { once: true });
  } else {
    startObservers();
  }
})();
</script>
"""


class OCRAnnotationSystem:
    def __init__(self) -> None:
        self.file_list: List[Path] = []
        self.current_index = 0
        self.results_data: List[Any] = []
        self.results_index_map: Dict[str, int] = {}
        self.current_record_index: Optional[int] = None
        self.original_data: Dict[str, Any] = {}
        self.current_data: Dict[str, Any] = {}
        self.extra_data: Dict[str, Any] = {}
        self.is_modified = False
        self.save_lock = Lock()
        self.last_save_time: Optional[str] = None
        self.pending_navigation: Optional[int] = None
        self.save_enabled = True
        self.status_level = "green"
        self.status_message = "Saved"
        self.results_error: Optional[str] = None
        self.viewer_serial = 0
        self.load_initial_state()

    def load_initial_state(self) -> None:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.scan_files()
        self.load_results_store()
        if self.file_list:
            self.load_current_file()
        else:
            self.original_data = self.create_default_record("")
            self.current_data = copy.deepcopy(self.original_data)
            self.extra_data = {}
            self.current_record_index = None
            if self.results_error:
                self.set_status("red", self.results_error)
            else:
                self.set_status("yellow", "No images found. Put jpg / jpeg / png files in images/.")

    def scan_files(self) -> None:
        files = [
            path
            for path in IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        self.file_list = sorted(files, key=lambda item: item.name.lower())
        if self.file_list:
            self.current_index = min(self.current_index, len(self.file_list) - 1)
        else:
            self.current_index = 0

    def load_results_store(self) -> None:
        self.results_data = []
        self.results_index_map = {}
        self.save_enabled = True
        self.results_error = None

        if not RESULTS_FILE.exists():
            return

        try:
            loaded = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            self.save_enabled = False
            self.results_error = f"Failed to parse results.json: {exc}"
            return

        if not isinstance(loaded, list):
            self.save_enabled = False
            self.results_error = "results.json must contain a top-level array. Saving has been disabled."
            return

        self.results_data = loaded
        duplicate_names: List[str] = []

        for index, record in enumerate(self.results_data):
            if not isinstance(record, dict):
                continue
            filename = record.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                continue
            normalized = filename.strip()
            if normalized in self.results_index_map:
                duplicate_names.append(normalized)
                continue
            self.results_index_map[normalized] = index

        if duplicate_names:
            repeated = ", ".join(sorted(set(duplicate_names))[:5])
            self.save_enabled = False
            self.results_error = f"results.json contains duplicate filename values: {repeated}"

    def create_default_record(self, filename: str) -> Dict[str, Any]:
        record: Dict[str, Any] = {}
        for field, config in SCHEMA_CONFIG.items():
            if field == "filename":
                record[field] = filename
            elif config["type"] == "number":
                record[field] = None
            else:
                record[field] = ""
        return record

    def normalize_value(self, field: str, value: Any) -> Any:
        config = SCHEMA_CONFIG[field]
        field_type = config["type"]
        if field == "filename":
            return self.current_filename()
        if field_type == "number":
            if value in ("", None):
                return None
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return value
        if value is None:
            return ""
        return str(value)

    def current_filename(self) -> str:
        if not self.file_list:
            return ""
        return self.file_list[self.current_index].name

    def current_file_path(self) -> Optional[Path]:
        if not self.file_list:
            return None
        return self.file_list[self.current_index]

    def load_current_file(self) -> None:
        filename = self.current_filename()
        record, record_index, extra_data, has_existing = self.load_record_for_image(filename)
        self.current_record_index = record_index
        self.original_data = copy.deepcopy(record)
        self.current_data = copy.deepcopy(record)
        self.extra_data = extra_data
        self.is_modified = False
        self.pending_navigation = None
        self.viewer_serial += 1

        if self.results_error:
            self.set_status("red", self.results_error)
        elif has_existing:
            self.set_status("green", "Loaded existing record")
        else:
            self.set_status("yellow", "No record exists for this image yet. Edits will be written to results.json.")

    def load_record_for_image(
        self, filename: str
    ) -> Tuple[Dict[str, Any], Optional[int], Dict[str, Any], bool]:
        record_index = self.results_index_map.get(filename)
        if record_index is None:
            default_record = self.create_default_record(filename)
            return default_record, None, {}, False

        record = self.results_data[record_index]
        if not isinstance(record, dict):
            default_record = self.create_default_record(filename)
            return default_record, None, {}, False

        merged = self.create_default_record(filename)
        for field in FIELD_ORDER:
            if field in record:
                merged[field] = record[field]
        merged["filename"] = filename

        extra_data = {
            key: value for key, value in record.items() if key not in SCHEMA_CONFIG
        }
        return merged, record_index, extra_data, True

    def set_status(self, level: str, message: str) -> None:
        self.status_level = level
        self.status_message = message

    def format_status_html(self) -> str:
        dot_class = {
            "green": "status-green",
            "yellow": "status-yellow",
            "red": "status-red",
        }.get(self.status_level, "status-green")
        save_time = self.last_save_time or "Not saved yet"
        return f"""
        <div class="status-host">
          <div class="status-row">
            <div class="status-main">
              <span class="status-dot {dot_class}"></span>
              <span class="status-label">{html.escape(self.status_message)}</span>
            </div>
            <span class="status-time">Last saved: {html.escape(save_time)}</span>
          </div>
        </div>
        """

    def render_confirm_message(self) -> str:
        if self.pending_navigation is None or not self.file_list:
            return '<div class="confirm-text">There are unsaved changes.</div>'
        target = self.pending_navigation + 1
        total = len(self.file_list)
        return (
            '<div class="confirm-text">'
            f"There are unsaved changes. Save or discard them before switching to image {target} / {total}?"
            "</div>"
        )

    def progress_text(self) -> str:
        if not self.file_list:
            return "0/0"
        return f"{self.current_index + 1}/{len(self.file_list)}"

    def compose_record_for_save(self) -> Dict[str, Any]:
        merged = copy.deepcopy(self.extra_data)
        for field in FIELD_ORDER:
            merged[field] = self.current_data.get(field)
        merged["filename"] = self.current_filename()
        return merged

    def atomic_write_results(self) -> None:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_path = RESULTS_FILE.with_suffix(".json.tmp")
        payload = json.dumps(self.results_data, ensure_ascii=False, indent=2)
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(RESULTS_FILE)

    def save_current_record(self, manual: bool) -> bool:
        with self.save_lock:
            if not self.file_list:
                self.set_status("yellow", "There is no image to save.")
                return False
            if not self.save_enabled:
                if self.results_error:
                    self.set_status("red", self.results_error)
                else:
                    self.set_status("red", "Saving is currently disabled.")
                return False
            if not self.is_modified and self.current_record_index is not None:
                if manual:
                    self.set_status("green", "There are no pending edits to save.")
                return False

            filename = self.current_filename()
            record = self.compose_record_for_save()
            self.set_status("yellow", "Saving...")

            try:
                target_index = self.results_index_map.get(filename)
                if target_index is None:
                    self.results_data.append(record)
                    target_index = len(self.results_data) - 1
                    self.results_index_map[filename] = target_index
                else:
                    existing = self.results_data[target_index]
                    if not isinstance(existing, dict):
                        raise ValueError("results.json contains a non-object record and cannot be updated safely.")
                    self.results_data[target_index] = record

                self.atomic_write_results()
                self.current_record_index = target_index
                self.extra_data = {
                    key: value for key, value in record.items() if key not in SCHEMA_CONFIG
                }
                self.original_data = copy.deepcopy(self.current_data)
                self.is_modified = False
                self.last_save_time = time.strftime("%Y-%m-%d %H:%M:%S")
                if manual:
                    self.set_status("green", "Saved manually")
                else:
                    self.set_status("green", "Saved automatically")
                return True
            except Exception as exc:
                self.set_status("red", f"Save failed: {exc}")
                return False

    def request_navigation(self, target_index: int) -> None:
        if not self.file_list:
            self.set_status("yellow", "There is no image to switch.")
            return
        if target_index < 0 or target_index >= len(self.file_list):
            self.set_status("red", "Target index is outside the current image range.")
            return
        if self.is_modified:
            self.pending_navigation = target_index
            self.set_status("yellow", "Unsaved changes detected. Confirm how to switch first.")
            return
        self.pending_navigation = None
        self.current_index = target_index
        self.load_current_file()

    def go_previous(self) -> None:
        if not self.file_list:
            self.set_status("yellow", "There is no image to switch.")
            return
        target = (self.current_index - 1) % len(self.file_list)
        self.request_navigation(target)

    def go_next(self) -> None:
        if not self.file_list:
            self.set_status("yellow", "There is no image to switch.")
            return
        target = (self.current_index + 1) % len(self.file_list)
        self.request_navigation(target)

    def jump_to(self, requested_index: Any) -> None:
        if not self.file_list:
            self.set_status("yellow", "There is no image to switch.")
            return
        if requested_index in (None, ""):
            self.set_status("red", "Enter the target image index.")
            return
        try:
            target = int(requested_index) - 1
        except Exception:
            self.set_status("red", "The target image index must be an integer.")
            return
        self.request_navigation(target)

    def confirm_save_and_switch(self) -> None:
        if self.pending_navigation is None:
            self.set_status("yellow", "There is no pending navigation target.")
            return
        saved = self.save_current_record(manual=True)
        if not saved:
            return
        self.current_index = self.pending_navigation
        self.pending_navigation = None
        self.load_current_file()

    def confirm_discard_and_switch(self) -> None:
        if self.pending_navigation is None:
            self.set_status("yellow", "There is no pending navigation target.")
            return
        self.current_index = self.pending_navigation
        self.pending_navigation = None
        self.load_current_file()
        self.set_status("green", "Discarded changes and switched images")

    def cancel_navigation(self) -> None:
        self.pending_navigation = None
        if self.is_modified:
            self.set_status("yellow", "Navigation canceled. Current changes are still unsaved.")
        else:
            self.set_status("green", "Navigation canceled")

    def on_field_change(self, field_name: str, value: Any) -> Tuple[str, str]:
        if field_name not in SCHEMA_CONFIG:
            return self.generate_diff_html(), self.format_status_html()

        with self.save_lock:
            normalized = self.normalize_value(field_name, value)
            previous = self.current_data.get(field_name)
            self.current_data[field_name] = normalized
            self.current_data["filename"] = self.current_filename()
            if previous != normalized:
                self.is_modified = True
                self.set_status("yellow", "Editing...")

        return self.generate_diff_html(), self.format_status_html()

    def reset_view(self) -> Tuple[str, str, str]:
        self.viewer_serial += 1
        return (
            self.render_viewer_html(),
            self.generate_diff_html(),
            self.format_status_html(),
        )

    def render_viewer_html(self) -> str:
        image_path = self.current_file_path()
        if image_path is None:
            return """
            <div class="viewer-host">
              <div class="viewer-shell">
                <div class="viewer-empty">
                  <div>
                    <div>No image is available to display.</div>
                    <div class="viewer-note">Put jpg / jpeg / png files in images/.</div>
                  </div>
                </div>
              </div>
            </div>
            """

        data_url, note = self.image_to_data_url(image_path)
        viewer_id = f"viewer-{self.viewer_serial}-{int(time.time() * 1000)}"
        escaped_data_url = html.escape(data_url, quote=True)
        escaped_filename = html.escape(image_path.name)
        escaped_note = html.escape(note)

        return f"""
        <div class="viewer-host">
          <div id="{viewer_id}" class="viewer-shell" data-ocr-viewer="1">
            <div class="viewer-header">
              <span>{escaped_filename}</span>
              <span class="viewer-note">{escaped_note}</span>
            </div>
            <div class="viewer-stage">
              <canvas class="viewer-canvas"></canvas>
              <img class="viewer-source" src="{escaped_data_url}" alt="{escaped_filename}" />
            </div>
          </div>
        </div>
        """

    def image_to_data_url(self, image_path: Path) -> Tuple[str, str]:
        note = "Use the mouse wheel to zoom and drag to pan."
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not read the image.")
            success, encoded = cv2.imencode(image_path.suffix.lower(), image)
            if not success:
                success, encoded = cv2.imencode(".png", image)
                if not success:
                    raise ValueError("Image encoding failed.")
                mime = "image/png"
            else:
                mime = (
                    "image/png"
                    if image_path.suffix.lower() == ".png"
                    else "image/jpeg"
                )
            return (
                f"data:{mime};base64,{base64.b64encode(encoded.tobytes()).decode('utf-8')}",
                note,
            )
        except Exception as exc:
            placeholder = self.create_placeholder_image(image_path.name, str(exc))
            buffer = BytesIO()
            placeholder.save(buffer, format="PNG")
            return (
                f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}",
                f"Failed to load image: {exc}",
            )

    def create_placeholder_image(self, filename: str, error_message: str) -> Image.Image:
        image = Image.new("RGB", (1280, 900), color=(246, 248, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 1200, 820), outline=(205, 214, 229), width=4)
        draw.text((120, 160), "Failed to load image", fill=(169, 62, 62))
        draw.text((120, 230), filename, fill=(70, 80, 96))
        draw.text((120, 300), error_message[:120], fill=(106, 118, 140))
        return image

    def diff_lines_for_text(self, before: str, after: str) -> List[str]:
        import difflib

        before_lines = before.splitlines() or [before]
        after_lines = after.splitlines() or [after]
        return list(difflib.ndiff(before_lines, after_lines))

    def generate_diff_html(self) -> str:
        changed_blocks: List[str] = []
        for field in FIELD_ORDER:
            before = self.original_data.get(field)
            after = self.current_data.get(field)
            if before == after:
                continue

            label = SCHEMA_CONFIG[field]["label"]
            block_lines = [
                f'<div class="diff-field"><div class="diff-title">Field: {html.escape(label)} ({html.escape(field)})</div>'
            ]

            before_text = "" if before is None else str(before)
            after_text = "" if after is None else str(after)

            if "\n" in before_text or "\n" in after_text:
                for line in self.diff_lines_for_text(before_text, after_text):
                    safe_line = html.escape(line[2:] if len(line) > 2 else line)
                    if line.startswith("- "):
                        block_lines.append(
                            f'<span class="diff-line del">- Original: {safe_line}</span>'
                        )
                    elif line.startswith("+ "):
                        block_lines.append(
                            f'<span class="diff-line add">+ Updated: {safe_line}</span>'
                        )
                    elif line.startswith("? "):
                        continue
                    else:
                        block_lines.append(
                            f'<span class="diff-line same">  Context: {safe_line}</span>'
                        )
            else:
                if before_text:
                    block_lines.append(
                        f'<span class="diff-line del">- Original: {html.escape(before_text)}</span>'
                    )
                if after_text:
                    block_lines.append(
                        f'<span class="diff-line add">+ Updated: {html.escape(after_text)}</span>'
                    )
                if not before_text and not after_text:
                    block_lines.append(
                        '<span class="diff-line same">  Context: empty</span>'
                    )

            block_lines.append("</div>")
            changed_blocks.append("".join(block_lines))

        if not changed_blocks:
            return """
            <div class="diff-host">
              <div class="diff-clean">No unsaved changes</div>
            </div>
            """

        return f"""
        <div class="diff-host">
          <div class="diff-block">
            {''.join(changed_blocks)}
          </div>
        </div>
        """

    def full_view_payload(self) -> List[Any]:
        return [
            self.render_viewer_html(),
            self.progress_text(),
            self.generate_diff_html(),
            self.format_status_html(),
            gr.update(visible=self.pending_navigation is not None),
            self.render_confirm_message(),
            *[gr.update(value=self.current_data.get(field)) for field in FIELD_ORDER],
        ]

    def save_payload(self) -> List[Any]:
        return [
            self.generate_diff_html(),
            self.format_status_html(),
            gr.update(visible=self.pending_navigation is not None),
            self.render_confirm_message(),
        ]


def build_ui() -> gr.Blocks:
    system = OCRAnnotationSystem()

    with gr.Blocks(
        title="OCR Annotation Review System",
        css=CUSTOM_CSS,
        head=APP_HEAD,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    ) as demo:
        with gr.Column(elem_classes="app-shell"):
            with gr.Row():
                btn_prev = gr.Button(
                    "← Previous (Left)",
                    elem_id="nav-prev-button",
                    interactive=bool(system.file_list),
                )
                btn_next = gr.Button(
                    "Next (Right) →",
                    elem_id="nav-next-button",
                    interactive=bool(system.file_list),
                )
                jump_input = gr.Number(
                    label="Jump To",
                    precision=0,
                    minimum=1,
                    interactive=bool(system.file_list),
                )
                btn_jump = gr.Button(
                    "Jump",
                    elem_id="jump-button",
                    interactive=bool(system.file_list),
                )
                progress_text = gr.Textbox(
                    label="Progress",
                    value=system.progress_text(),
                    interactive=False,
                )

            with gr.Row():
                with gr.Column(scale=6):
                    html_viewer = gr.HTML(
                        value=system.render_viewer_html(),
                        label="Image Viewer",
                    )
                    btn_reset_view = gr.Button("Reset View", elem_id="reset-view-button")

                with gr.Column(scale=4):
                    form_components: Dict[str, gr.components.Component] = {}
                    for field, config in SCHEMA_CONFIG.items():
                        elem_classes = ["read-only-field"] if not config["editable"] else None
                        common_kwargs = {
                            "label": config["label"],
                            "info": config.get("description"),
                            "interactive": config["editable"],
                            "value": system.current_data.get(field),
                            "elem_id": f"field-{field}",
                            "elem_classes": elem_classes,
                        }

                        if config["type"] == "text":
                            component = gr.Textbox(lines=1, **common_kwargs)
                        elif config["type"] == "textarea":
                            component = gr.Textbox(
                                lines=config.get("lines", 5),
                                **common_kwargs,
                            )
                        elif config["type"] == "number":
                            component = gr.Number(
                                precision=0
                                if field in {"idx", "sub_question_count"}
                                else None,
                                **common_kwargs,
                            )
                        elif config["type"] == "dropdown":
                            component = gr.Dropdown(
                                choices=config.get("choices", []),
                                allow_custom_value=False,
                                **common_kwargs,
                            )
                        else:
                            raise ValueError(f"Unsupported field type: {config['type']}")

                        form_components[field] = component

                    status_indicator = gr.HTML(
                        value=system.format_status_html(),
                        label="Status",
                    )

            with gr.Group(
                visible=False,
                elem_classes="confirm-host",
            ) as confirm_panel:
                gr.HTML("<div style='font-weight:700;color:#6b531d;'>Unsaved Changes</div>")
                confirm_message = gr.HTML(value=system.render_confirm_message())
                with gr.Row():
                    btn_confirm_save = gr.Button("Save and Switch", elem_id="confirm-save-button")
                    btn_confirm_discard = gr.Button(
                        "Discard and Switch",
                        elem_id="confirm-discard-button",
                    )
                    btn_confirm_cancel = gr.Button("Cancel", elem_id="confirm-cancel-button")

            diff_viewer = gr.HTML(
                value=system.generate_diff_html(),
                label="Change Diff",
            )

            hidden_auto_save_button = gr.Button(
                "auto-save",
                elem_id="hidden-auto-save-button",
                elem_classes="hidden-action",
            )
            hidden_manual_save_button = gr.Button(
                "manual-save",
                elem_id="hidden-manual-save-button",
                elem_classes="hidden-action",
            )

        ordered_components = [form_components[field] for field in FIELD_ORDER]
        full_outputs = [
            html_viewer,
            progress_text,
            diff_viewer,
            status_indicator,
            confirm_panel,
            confirm_message,
            *ordered_components,
        ]
        save_outputs = [diff_viewer, status_indicator, confirm_panel, confirm_message]

        def go_prev_handler() -> List[Any]:
            system.go_previous()
            return system.full_view_payload()

        def go_next_handler() -> List[Any]:
            system.go_next()
            return system.full_view_payload()

        def jump_handler(requested_index: Any) -> List[Any]:
            system.jump_to(requested_index)
            return system.full_view_payload()

        def confirm_save_switch_handler() -> List[Any]:
            system.confirm_save_and_switch()
            return system.full_view_payload()

        def confirm_discard_switch_handler() -> List[Any]:
            system.confirm_discard_and_switch()
            return system.full_view_payload()

        def cancel_navigation_handler() -> List[Any]:
            system.cancel_navigation()
            return system.full_view_payload()

        def auto_save_handler() -> List[Any]:
            system.save_current_record(manual=False)
            return system.save_payload()

        def manual_save_handler() -> List[Any]:
            system.save_current_record(manual=True)
            return system.save_payload()

        btn_prev.click(fn=go_prev_handler, outputs=full_outputs)
        btn_next.click(fn=go_next_handler, outputs=full_outputs)
        btn_jump.click(fn=jump_handler, inputs=jump_input, outputs=full_outputs)
        btn_confirm_save.click(fn=confirm_save_switch_handler, outputs=full_outputs)
        btn_confirm_discard.click(
            fn=confirm_discard_switch_handler,
            outputs=full_outputs,
        )
        btn_confirm_cancel.click(fn=cancel_navigation_handler, outputs=full_outputs)
        hidden_auto_save_button.click(fn=auto_save_handler, outputs=save_outputs)
        hidden_manual_save_button.click(fn=manual_save_handler, outputs=save_outputs)
        btn_reset_view.click(
            fn=system.reset_view,
            outputs=[html_viewer, diff_viewer, status_indicator],
        )

        for field in FIELD_ORDER:
            component = form_components[field]
            if not SCHEMA_CONFIG[field]["editable"]:
                continue

            def change_handler(value: Any, field_name: str = field) -> Tuple[str, str]:
                return system.on_field_change(field_name, value)

            if SCHEMA_CONFIG[field]["type"] in {"text", "textarea"}:
                component.input(
                    fn=change_handler,
                    inputs=component,
                    outputs=[diff_viewer, status_indicator],
                )
            else:
                component.change(
                    fn=change_handler,
                    inputs=component,
                    outputs=[diff_viewer, status_indicator],
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
