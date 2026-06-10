# -*- coding: utf-8 -*-
"""
PyQt6 + DeepFace single-file local face search tool.
Install:
    pip install PyQt6 deepface opencv-python numpy flask
Run:
    python face_search_pyqt6_single_advanced_api_english.py
Design goals:
    1. One .py file only: GUI, worker thread, recognition engine, CSV/JSON export.
    2. High-accuracy default pipeline: DeepFace model "Buffalo_L" first, RetinaFace detector,
       face alignment enabled, cosine distance. If "Automatic Highest Accuracy" is selected and
       Buffalo_L is unavailable in the local DeepFace build, the engine can fall back to ArcFace.
    3. Responsive UI: all expensive image processing runs in QThread. The main thread only renders
       batched table updates, progress, image previews and messages.
    4. Result rule: if there are threshold matches, show all matched faces; do not cut by Top-K.
       If there are no threshold matches, show the nearest fallback candidates.
    5. Eye-friendly UI colors: every explicit RGB color in code and stylesheet uses blue channel = 0.
       Preview images are also rendered with their displayed blue channel set to zero. Original files
       are never modified.
Important note:
    Face similarity is not a legal or biometric certainty. The displayed similarity_percent is a
    deterministic display score derived from the selected distance metric. It is not a probability.
"""
from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

from PyQt6.QtCore import QEvent, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "High-Accuracy Face Search"
APP_TITLE = "High-Accuracy Local Face Search"
APP_DIR = Path(__file__).resolve().parent
LOG_FILE = APP_DIR / "face_search_pyqt6.log"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# DeepFace model names are case-sensitive. This app exposes the high-accuracy options first.
AUTO_HIGH_ACCURACY_MODEL = "Automatic Highest Accuracy"
PRIMARY_HIGH_ACCURACY_MODEL = "Buffalo_L"
FALLBACK_HIGH_ACCURACY_MODEL = "ArcFace"
DEFAULT_MODEL_NAME = AUTO_HIGH_ACCURACY_MODEL
DEFAULT_DETECTOR_BACKEND = "retinaface"
DEFAULT_DISTANCE_METRIC = "cosine"
DEFAULT_NORMALIZATION = "base"
DEFAULT_MANUAL_THRESHOLD = 0.68
DEFAULT_FALLBACK_CANDIDATES = 20
DEFAULT_MAX_LIVE_ROWS = 2500

# Every explicit color below is RGB with blue channel = 0.
COLOR_TEXT = QColor(0, 0, 0)
COLOR_WINDOW = QColor(224, 232, 0)
COLOR_PANEL = QColor(232, 236, 0)
COLOR_FIELD = QColor(248, 248, 0)
COLOR_HEADER = QColor(192, 208, 0)
COLOR_BUTTON = QColor(176, 192, 0)
COLOR_PRIMARY = QColor(160, 208, 0)
COLOR_BORDER = QColor(96, 112, 0)
COLOR_MATCH = QColor(208, 232, 0)
COLOR_CANDIDATE = QColor(248, 248, 0)
COLOR_ERROR = QColor(232, 176, 0)
COLOR_HIGHLIGHT = QColor(128, 144, 0)


logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)

# DeepFace/TensorFlow model execution is serialized across GUI and API requests.
# This avoids simultaneous model loading/inference from multiple threads in the same Python process.
RECOGNITION_LOCK = threading.RLock()


@dataclass
class FaceSearchConfig:
    """Configuration transferred from the GUI to the worker thread."""

    target_image_path: str
    search_directory_path: str
    output_parent_directory: str
    model_name: str = DEFAULT_MODEL_NAME
    detector_backend: str = DEFAULT_DETECTOR_BACKEND
    distance_metric: str = DEFAULT_DISTANCE_METRIC
    threshold: float = DEFAULT_MANUAL_THRESHOLD
    use_pretrained_threshold: bool = True
    fallback_candidates: int = DEFAULT_FALLBACK_CANDIDATES
    enforce_detection: bool = True
    align_faces: bool = True
    expand_percentage: int = 0
    normalization: str = DEFAULT_NORMALIZATION
    minimum_face_confidence: float = 0.0
    recursive_search: bool = True
    copy_selected_images: bool = True
    save_all_ranked_csv: bool = True
    save_summary_json: bool = True
    auto_open_output_directory: bool = False
    max_live_rows: int = DEFAULT_MAX_LIVE_ROWS
    allow_model_fallback: bool = True


@dataclass
class FaceComparisonRecord:
    """One comparison result for one detected face in one image."""

    rank: int
    image_path: str
    face_index: int
    distance: float
    similarity_score: float
    similarity_percent: float
    is_match: bool
    threshold: float
    result_mode: str
    model_name: str
    detector_backend: str
    distance_metric: str
    face_confidence: Optional[float]
    facial_area_json: str
    error_message: str = ""

    def to_csv_row(self) -> List[str]:
        return [
            str(self.rank),
            "yes" if self.is_match else "no",
            self.result_mode,
            f"{self.distance:.8f}",
            f"{self.similarity_score:.8f}",
            f"{self.similarity_percent:.2f}%",
            f"{self.threshold:.8f}",
            self.model_name,
            self.detector_backend,
            self.distance_metric,
            self.image_path,
            str(self.face_index),
            "" if self.face_confidence is None else f"{self.face_confidence:.8f}",
            self.facial_area_json,
            self.error_message,
        ]


@dataclass
class FaceSearchSummary:
    """Structured final task summary written to JSON and displayed in the GUI."""

    started_at: str
    finished_at: str
    target_image_path: str
    search_directory_path: str
    output_directory_path: str
    requested_model_name: str
    actual_model_name: str
    detector_backend: str
    distance_metric: str
    threshold: float
    threshold_source: str
    fallback_candidates: int
    image_files_total: int
    image_files_processed: int
    faces_compared: int
    files_without_detected_faces: int
    files_failed: int
    threshold_matches_count: int
    selected_results_count: int
    result_mode: str
    best_distance: Optional[float]
    best_similarity_percent: Optional[float]


class FaceRecognitionEngine:
    """
    Pure recognition/search logic.

    The GUI deliberately does not call DeepFace directly. Keeping recognition code here makes it
    easier to test, replace the model, or reuse the same engine from another UI later.
    """

    def __init__(self, config: FaceSearchConfig) -> None:
        self.config = config
        self._deepface = None
        self.actual_model_name = self.resolve_model_name(config.model_name)
        self.threshold_source = "manual"
        self.effective_threshold = float(config.threshold)

    @staticmethod
    def resolve_model_name(requested_model_name: str) -> str:
        """Map the GUI's automatic option to the strongest preferred model name."""
        if requested_model_name == AUTO_HIGH_ACCURACY_MODEL:
            return PRIMARY_HIGH_ACCURACY_MODEL
        return requested_model_name

    def load_deepface(self):
        """Lazy import DeepFace so the window appears before TensorFlow/model loading starts."""
        if self._deepface is None:
            from deepface import DeepFace

            self._deepface = DeepFace
        return self._deepface

    @staticmethod
    def iter_image_files(directory_path: str, recursive: bool) -> List[str]:
        """Return supported image files in deterministic order."""
        root = Path(directory_path)
        if not root.exists() or not root.is_dir():
            return []

        iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
        image_paths = [
            str(path)
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        image_paths.sort(key=lambda value: value.lower())
        return image_paths

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= 0 or not math.isfinite(float(norm)):
            return vector
        return vector / norm

    @classmethod
    def calculate_distance(cls, left_embedding: Sequence[float], right_embedding: Sequence[float], metric: str) -> float:
        """
        Calculate a distance where lower is more similar.

        Supported metrics mirror the common DeepFace choices. Cosine is the default because it is
        stable for embedding comparison and easy to interpret: cosine_distance = 1 - cosine_similarity.
        """
        left = np.asarray(left_embedding, dtype=np.float64)
        right = np.asarray(right_embedding, dtype=np.float64)

        if left.size == 0 or right.size == 0 or left.shape != right.shape:
            return float("inf")

        if metric == "cosine":
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)
            if left_norm <= 0 or right_norm <= 0:
                return 1.0
            similarity = float(np.dot(left, right) / (left_norm * right_norm))
            distance = 1.0 - similarity
        elif metric == "euclidean":
            distance = float(np.linalg.norm(left - right))
        elif metric == "euclidean_l2":
            distance = float(np.linalg.norm(cls._l2_normalize(left) - cls._l2_normalize(right)))
        elif metric == "angular":
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)
            if left_norm <= 0 or right_norm <= 0:
                return 1.0
            cosine_value = float(np.dot(left, right) / (left_norm * right_norm))
            cosine_value = max(-1.0, min(1.0, cosine_value))
            distance = float(math.acos(cosine_value) / math.pi)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

        if not math.isfinite(distance):
            return float("inf")
        return round(distance, 8)

    @staticmethod
    def distance_to_similarity(distance: float, metric: str) -> Tuple[float, float]:
        """
        Convert distance to a display-only similarity score.

        For cosine/angular distances this is direct and intuitive. For Euclidean distances, a bounded
        monotonic transform is used only for UI readability. It is not a probability.
        """
        if not math.isfinite(distance):
            return 0.0, 0.0

        if metric in {"cosine", "angular"}:
            similarity_score = 1.0 - distance
        else:
            similarity_score = 1.0 / (1.0 + max(0.0, distance))

        similarity_score = max(0.0, min(1.0, similarity_score))
        return round(similarity_score, 8), round(similarity_score * 100.0, 2)

    @staticmethod
    def safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:  # noqa: BLE001
            return "{}"

    @staticmethod
    def extract_face_confidence(face_object: Dict[str, Any]) -> Optional[float]:
        raw_value = face_object.get("face_confidence", face_object.get("confidence"))
        if raw_value is None:
            return None
        try:
            return float(raw_value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def select_primary_face(face_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pick the target face with the strongest signal.

        If the target image contains multiple faces, using the largest/highest-confidence detection is
        more predictable than blindly taking the first list element.
        """
        if not face_objects:
            raise ValueError("No face object to select.")

        def face_score(face_object: Dict[str, Any]) -> float:
            area = face_object.get("facial_area", {}) or {}
            width = float(area.get("w", 0) or 0)
            height = float(area.get("h", 0) or 0)
            confidence = FaceRecognitionEngine.extract_face_confidence(face_object)
            confidence_value = 1.0 if confidence is None else max(0.0, confidence)
            return width * height * confidence_value

        return max(face_objects, key=face_score)

    def _call_represent_once(self, image_path: str, model_name: str) -> List[Dict[str, Any]]:
        """Call DeepFace.represent while remaining compatible with older DeepFace versions."""
        DeepFace = self.load_deepface()

        requested_arguments = {
            "img_path": image_path,
            "model_name": model_name,
            "detector_backend": self.config.detector_backend,
            "enforce_detection": self.config.enforce_detection,
            "align": self.config.align_faces,
            "normalization": self.config.normalization,
            "expand_percentage": self.config.expand_percentage,
        }

        # Recent DeepFace versions expose more parameters than older versions. Filtering by signature
        # keeps the app from crashing when a user has a slightly different installed version.
        try:
            signature = inspect.signature(DeepFace.represent)
            supported_arguments = {
                key: value for key, value in requested_arguments.items() if key in signature.parameters
            }
        except Exception:  # noqa: BLE001
            supported_arguments = requested_arguments

        # DeepFace model loading and inference are protected by a process-wide lock because the
        # GUI worker and the optional Flask API can be active at the same time. This keeps the
        # application stable at the cost of serializing recognition calls.
        with RECOGNITION_LOCK:
            result = DeepFace.represent(**supported_arguments)

        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return []

    def extract_face_embeddings(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract all face embeddings from one image, using fallback only when configured."""
        try:
            return self._call_represent_once(image_path, self.actual_model_name)
        except Exception as primary_error:  # noqa: BLE001
            # Automatic mode may attempt Buffalo_L first and fall back to ArcFace when Buffalo_L is not
            # available in the local DeepFace installation. The exception is logged and surfaced by the GUI.
            requested_automatic = self.config.model_name == AUTO_HIGH_ACCURACY_MODEL
            if requested_automatic and self.config.allow_model_fallback and self.actual_model_name != FALLBACK_HIGH_ACCURACY_MODEL:
                logging.warning(
                    "Primary model %s failed for %s. Falling back to %s. Error: %s",
                    self.actual_model_name,
                    image_path,
                    FALLBACK_HIGH_ACCURACY_MODEL,
                    primary_error,
                )
                self.actual_model_name = FALLBACK_HIGH_ACCURACY_MODEL
                return self._call_represent_once(image_path, self.actual_model_name)
            raise

    def load_threshold(self) -> Tuple[float, str]:
        """
        Prefer DeepFace's pre-tuned threshold when requested and available.

        If the threshold lookup is not available for a model/metric pair, the GUI-provided manual
        threshold is used. This avoids hard failure while making the threshold source explicit.
        """
        if not self.config.use_pretrained_threshold:
            self.effective_threshold = float(self.config.threshold)
            self.threshold_source = "manual"
            return self.effective_threshold, self.threshold_source

        try:
            from deepface.modules import verification

            threshold = float(verification.find_threshold(self.actual_model_name, self.config.distance_metric))
            if math.isfinite(threshold) and threshold > 0:
                self.effective_threshold = threshold
                self.threshold_source = "deepface_pretrained"
                return self.effective_threshold, self.threshold_source
        except Exception as exc:  # noqa: BLE001
            logging.info("Could not load DeepFace threshold, using manual value. Error: %s", exc)

        self.effective_threshold = float(self.config.threshold)
        self.threshold_source = "manual_fallback"
        return self.effective_threshold, self.threshold_source

    def get_target_embedding(self) -> Tuple[Sequence[float], Dict[str, Any], int]:
        """Extract the primary target face embedding and return the number of faces found."""
        face_objects = self.extract_face_embeddings(self.config.target_image_path)
        if not face_objects:
            raise RuntimeError("No usable face was detected in the target image. Please choose a clearer, frontal, unobstructed target image.")

        primary_face = self.select_primary_face(face_objects)
        embedding = primary_face.get("embedding")
        if embedding is None:
            raise RuntimeError("A face was detected in the target image, but no embedding was generated.")
        return embedding, primary_face, len(face_objects)

    def compare_image_to_target(self, target_embedding: Sequence[float], image_path: str) -> List[FaceComparisonRecord]:
        """Compare every detected face in one image against the target embedding."""
        face_objects = self.extract_face_embeddings(image_path)
        comparison_records: List[FaceComparisonRecord] = []

        for face_index, face_object in enumerate(face_objects):
            embedding = face_object.get("embedding")
            if embedding is None:
                continue

            confidence = self.extract_face_confidence(face_object)
            if confidence is not None and confidence < self.config.minimum_face_confidence:
                continue

            distance = self.calculate_distance(target_embedding, embedding, self.config.distance_metric)
            similarity_score, similarity_percent = self.distance_to_similarity(distance, self.config.distance_metric)
            is_match = distance <= self.effective_threshold

            comparison_records.append(
                FaceComparisonRecord(
                    rank=0,
                    image_path=image_path,
                    face_index=face_index,
                    distance=distance,
                    similarity_score=similarity_score,
                    similarity_percent=similarity_percent,
                    is_match=is_match,
                    threshold=self.effective_threshold,
                    result_mode="live_candidate",
                    model_name=self.actual_model_name,
                    detector_backend=self.config.detector_backend,
                    distance_metric=self.config.distance_metric,
                    face_confidence=confidence,
                    facial_area_json=self.safe_json(face_object.get("facial_area", {})),
                )
            )

        return comparison_records

    @staticmethod
    def finalize_records(
        all_records: List[FaceComparisonRecord],
        threshold: float,
        fallback_candidates: int,
    ) -> Tuple[List[FaceComparisonRecord], List[FaceComparisonRecord], str]:
        """
        Apply the required final result rule.

        If threshold matches exist, return all of them. If none exist, return nearest fallback
        candidates. all_ranked_records always contains every comparable face in sorted order.
        """
        all_ranked_records = sorted(
            all_records,
            key=lambda record: (record.distance, record.image_path.lower(), record.face_index),
        )

        for rank, record in enumerate(all_ranked_records, start=1):
            record.rank = rank
            record.is_match = record.distance <= threshold

        threshold_matches = [record for record in all_ranked_records if record.is_match]
        if threshold_matches:
            for record in threshold_matches:
                record.result_mode = "threshold_match"
            return threshold_matches, all_ranked_records, "threshold_match_all"

        fallback_count = max(1, int(fallback_candidates))
        fallback_records = all_ranked_records[:fallback_count]
        for record in fallback_records:
            record.result_mode = "no_match_nearest_fallback"
        return fallback_records, all_ranked_records, "no_match_nearest_fallback"

    @staticmethod
    def create_output_directory(parent_directory: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = Path(parent_directory) / f"face_search_results_{timestamp}"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    @staticmethod
    def write_records_csv(csv_path: Path, records: List[FaceComparisonRecord]) -> None:
        headers = [
            "rank",
            "is_match",
            "result_mode",
            "distance_lower_is_better",
            "similarity_score_higher_is_better",
            "similarity_percent_higher_is_better",
            "threshold",
            "model_name",
            "detector_backend",
            "distance_metric",
            "image_path",
            "face_index",
            "face_confidence",
            "facial_area_json",
            "error_message",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(headers)
            for record in records:
                writer.writerow(record.to_csv_row())

    @staticmethod
    def write_summary_json(json_path: Path, summary: FaceSearchSummary) -> None:
        with json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(asdict(summary), file_handle, ensure_ascii=False, indent=2)

    @staticmethod
    def copy_selected_images(records: List[FaceComparisonRecord], output_directory: Path) -> None:
        """Copy final selected original images, preserving the source files untouched."""
        selected_directory = output_directory / "selected_images"
        selected_directory.mkdir(parents=True, exist_ok=True)

        used_names: Dict[str, int] = {}
        for record in records:
            source_path = Path(record.image_path)
            if not source_path.exists():
                continue

            original_name = source_path.name
            used_names[original_name] = used_names.get(original_name, 0) + 1
            duplicate_index = used_names[original_name]
            safe_stem = sanitize_filename(source_path.stem)
            safe_suffix = source_path.suffix.lower()
            destination_name = f"rank_{record.rank:05d}_face_{record.face_index}_{safe_stem}{safe_suffix}"
            if duplicate_index > 1:
                destination_name = f"rank_{record.rank:05d}_face_{record.face_index}_{safe_stem}_{duplicate_index}{safe_suffix}"

            try:
                shutil.copy2(source_path, selected_directory / destination_name)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not copy selected image %s: %s", source_path, exc)


class FaceSearchWorker(QThread):
    """Background worker that keeps DeepFace inference away from the GUI thread."""

    progress_changed = pyqtSignal(int, int, str)
    statistics_changed = pyqtSignal(dict)
    live_records_ready = pyqtSignal(list)
    message_ready = pyqtSignal(str)
    task_succeeded = pyqtSignal(dict, list, list)
    task_failed = pyqtSignal(str)

    def __init__(self, config: FaceSearchConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        started_at = datetime.now().isoformat(timespec="seconds")
        processed_image_count = 0
        files_without_faces = 0
        failed_file_count = 0
        faces_compared_count = 0
        all_records: List[FaceComparisonRecord] = []

        try:
            engine = FaceRecognitionEngine(self.config)
            self.message_ready.emit("Loading the high-accuracy face recognition model. The first run may need to download model weights.")

            target_embedding, _target_face, target_face_count = engine.get_target_embedding()
            if target_face_count > 1:
                self.message_ready.emit(f"{target_face_count} faces were detected in the target image. The app selected the face with the best combined area and confidence.")

            effective_threshold, threshold_source = engine.load_threshold()
            self.message_ready.emit(
                f"Actual model: {engine.actual_model_name} | Detector: {self.config.detector_backend} | "
                f"Distance metric: {self.config.distance_metric} | Threshold: {effective_threshold:.6f} ({threshold_source})"
            )

            image_files = engine.iter_image_files(self.config.search_directory_path, self.config.recursive_search)
            target_absolute_path = str(Path(self.config.target_image_path).resolve())
            image_files = [path for path in image_files if str(Path(path).resolve()) != target_absolute_path]
            total_image_count = len(image_files)

            if total_image_count == 0:
                raise RuntimeError("No supported image files were found in the search directory.")

            self.statistics_changed.emit(
                build_statistics_payload(
                    total_images=total_image_count,
                    processed_images=0,
                    faces_compared=0,
                    matches_count=0,
                    files_without_faces=0,
                    failed_files=0,
                    best_record=None,
                )
            )
            self.message_ready.emit(f"Found {total_image_count} images. Starting real-time image-by-image comparison.")

            live_batch: List[dict] = []
            last_batch_size = 0

            for image_path in image_files:
                if self.isInterruptionRequested():
                    self.message_ready.emit("Stop request received. The app is organizing the comparisons already completed.")
                    break

                try:
                    records = engine.compare_image_to_target(target_embedding, image_path)
                    processed_image_count += 1

                    if records:
                        for record in records:
                            all_records.append(record)
                            faces_compared_count += 1
                            live_batch.append(asdict(record))
                    else:
                        files_without_faces += 1

                    # Emit frequently enough for real-time feedback, but batch enough to avoid GUI jank.
                    if len(live_batch) - last_batch_size >= 12:
                        self.live_records_ready.emit(live_batch[last_batch_size:])
                        last_batch_size = len(live_batch)

                except Exception as exc:  # noqa: BLE001
                    processed_image_count += 1
                    failed_file_count += 1
                    logging.warning("Failed to process image %s\n%s", image_path, traceback.format_exc())
                    self.message_ready.emit(f"Skipped failed image: {Path(image_path).name} | {exc}")

                if live_batch and last_batch_size < len(live_batch):
                    self.live_records_ready.emit(live_batch[last_batch_size:])
                    last_batch_size = len(live_batch)

                best_record = min(all_records, key=lambda item: item.distance) if all_records else None
                matches_so_far = sum(1 for item in all_records if item.distance <= effective_threshold)
                self.progress_changed.emit(processed_image_count, total_image_count, image_path)
                self.statistics_changed.emit(
                    build_statistics_payload(
                        total_images=total_image_count,
                        processed_images=processed_image_count,
                        faces_compared=faces_compared_count,
                        matches_count=matches_so_far,
                        files_without_faces=files_without_faces,
                        failed_files=failed_file_count,
                        best_record=best_record,
                    )
                )

            if not all_records:
                raise RuntimeError("The search finished, but no comparable faces were detected. Check image quality or reduce detection strictness.")

            selected_records, all_ranked_records, result_mode = engine.finalize_records(
                all_records=all_records,
                threshold=effective_threshold,
                fallback_candidates=self.config.fallback_candidates,
            )

            output_directory = engine.create_output_directory(self.config.output_parent_directory)
            engine.write_records_csv(output_directory / "selected_results.csv", selected_records)
            if self.config.save_all_ranked_csv:
                engine.write_records_csv(output_directory / "all_faces_ranked.csv", all_ranked_records)
            if self.config.copy_selected_images:
                engine.copy_selected_images(selected_records, output_directory)

            threshold_matches_count = len([record for record in all_ranked_records if record.is_match])
            best_record = all_ranked_records[0] if all_ranked_records else None
            summary = FaceSearchSummary(
                started_at=started_at,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                target_image_path=self.config.target_image_path,
                search_directory_path=self.config.search_directory_path,
                output_directory_path=str(output_directory),
                requested_model_name=self.config.model_name,
                actual_model_name=engine.actual_model_name,
                detector_backend=self.config.detector_backend,
                distance_metric=self.config.distance_metric,
                threshold=effective_threshold,
                threshold_source=threshold_source,
                fallback_candidates=self.config.fallback_candidates,
                image_files_total=total_image_count,
                image_files_processed=processed_image_count,
                faces_compared=faces_compared_count,
                files_without_detected_faces=files_without_faces,
                files_failed=failed_file_count,
                threshold_matches_count=threshold_matches_count,
                selected_results_count=len(selected_records),
                result_mode=result_mode,
                best_distance=None if best_record is None else best_record.distance,
                best_similarity_percent=None if best_record is None else best_record.similarity_percent,
            )

            if self.config.save_summary_json:
                engine.write_summary_json(output_directory / "summary.json", summary)

            self.task_succeeded.emit(
                asdict(summary),
                [asdict(record) for record in selected_records],
                [asdict(record) for record in all_ranked_records],
            )

        except Exception as exc:  # noqa: BLE001
            logging.error("Face search task failed: %s\n%s", exc, traceback.format_exc())
            self.task_failed.emit(str(exc))


def build_statistics_payload(
    total_images: int,
    processed_images: int,
    faces_compared: int,
    matches_count: int,
    files_without_faces: int,
    failed_files: int,
    best_record: Optional[FaceComparisonRecord],
) -> Dict[str, Any]:
    return {
        "total_images": total_images,
        "processed_images": processed_images,
        "faces_compared": faces_compared,
        "matches_count": matches_count,
        "files_without_faces": files_without_faces,
        "failed_files": failed_files,
        "best_distance": None if best_record is None else best_record.distance,
        "best_similarity_percent": None if best_record is None else best_record.similarity_percent,
        "best_image_path": None if best_record is None else best_record.image_path,
    }


class FlaskApiServerThread(QThread):
    """
    Optional local Flask API server controlled by the GUI.

    The server is intentionally hosted inside a QThread so that starting/stopping the API does not
    block the PyQt event loop. Recognition work is delegated to FaceRecognitionEngine, exactly like
    the desktop search worker, so the API and GUI share one implementation path.
    """

    api_started = pyqtSignal(str)
    api_stopped = pyqtSignal()
    api_failed = pyqtSignal(str)
    api_message = pyqtSignal(str)

    def __init__(self, host: str, port: int, default_config: FaceSearchConfig) -> None:
        super().__init__()
        self.host = host.strip() or "127.0.0.1"
        self.port = int(port)
        self.default_config = default_config
        self._server = None

    def run(self) -> None:  # noqa: C901, PLR0915
        try:
            from flask import Flask, jsonify, request
            from werkzeug.serving import make_server
        except Exception as exc:  # noqa: BLE001
            self.api_failed.emit(
                "Flask is not available. Please run: pip install flask\n"
                f"Original error: {exc}"
            )
            return

        flask_app = Flask(__name__)

        @flask_app.get("/api/health")
        def health_check():
            return jsonify({
                "ok": True,
                "service": APP_NAME,
                "message": "face comparison api is running",
                "default_model_name": self.default_config.model_name,
                "default_detector_backend": self.default_config.detector_backend,
                "default_distance_metric": self.default_config.distance_metric,
            })

        @flask_app.post("/api/compare")
        def compare_faces():
            """
            Compare one target face with one candidate image.

            Supported request styles:
            1. JSON body with local paths:
               {"target_image_path": "...", "candidate_image_path": "..."}
               If target_image_path is omitted, the target selected in the GUI is used.
            2. multipart/form-data:
               target_image=<file>, candidate_image=<file>
               target_image may be omitted when the GUI already has a valid target path.
            """
            try:
                payload = parse_api_payload(request)
                with tempfile.TemporaryDirectory(prefix="face_api_compare_") as temporary_directory:
                    target_image_path = resolve_api_image_path(
                        request=request,
                        payload=payload,
                        field_name="target_image",
                        json_key="target_image_path",
                        fallback_path=self.default_config.target_image_path,
                        temporary_directory=temporary_directory,
                    )
                    candidate_image_path = resolve_api_image_path(
                        request=request,
                        payload=payload,
                        field_name="candidate_image",
                        json_key="candidate_image_path",
                        fallback_path="",
                        temporary_directory=temporary_directory,
                    )
                    if not candidate_image_path:
                        candidate_image_path = str(payload.get("image_path", "")).strip()

                    validate_existing_file(target_image_path, "target_image_path")
                    validate_existing_file(candidate_image_path, "candidate_image_path")

                    config = build_api_config_from_payload(
                        base_config=self.default_config,
                        payload=payload,
                        target_image_path=target_image_path,
                        search_directory_path=str(Path(candidate_image_path).parent),
                        output_parent_directory=str(Path(candidate_image_path).parent),
                    )
                    response_payload = run_api_compare(config, candidate_image_path)
                    return jsonify(response_payload)
            except Exception as exc:  # noqa: BLE001
                logging.warning("API compare failed: %s\n%s", exc, traceback.format_exc())
                return jsonify({"ok": False, "error": str(exc)}), 400

        @flask_app.post("/api/search")
        def search_directory():
            """
            Search a local directory through POST.

            This endpoint is synchronous: it returns after the search is complete. For very large
            directories, the GUI workflow is still better because it displays real-time progress.
            """
            try:
                payload = parse_api_payload(request)
                with tempfile.TemporaryDirectory(prefix="face_api_search_") as temporary_directory:
                    target_image_path = resolve_api_image_path(
                        request=request,
                        payload=payload,
                        field_name="target_image",
                        json_key="target_image_path",
                        fallback_path=self.default_config.target_image_path,
                        temporary_directory=temporary_directory,
                    )
                    search_directory_path = str(
                        payload.get("search_directory_path")
                        or payload.get("directory_path")
                        or self.default_config.search_directory_path
                        or ""
                    ).strip()
                    output_parent_directory = str(
                        payload.get("output_parent_directory")
                        or payload.get("output_directory_path")
                        or self.default_config.output_parent_directory
                        or search_directory_path
                    ).strip()

                    validate_existing_file(target_image_path, "target_image_path")
                    validate_existing_directory(search_directory_path, "search_directory_path")
                    if output_parent_directory:
                        Path(output_parent_directory).mkdir(parents=True, exist_ok=True)

                    config = build_api_config_from_payload(
                        base_config=self.default_config,
                        payload=payload,
                        target_image_path=target_image_path,
                        search_directory_path=search_directory_path,
                        output_parent_directory=output_parent_directory,
                    )
                    include_all_ranked = bool(payload.get("include_all_ranked", False))
                    response_payload = run_api_directory_search(config, include_all_ranked=include_all_ranked)
                    return jsonify(response_payload)
            except Exception as exc:  # noqa: BLE001
                logging.warning("API search failed: %s\n%s", exc, traceback.format_exc())
                return jsonify({"ok": False, "error": str(exc)}), 400

        try:
            # threaded=True allows the health endpoint to answer while a long comparison is running;
            # actual recognition calls are still serialized by RECOGNITION_LOCK.
            self._server = make_server(self.host, self.port, flask_app, threaded=True)
            url = f"http://{self.host}:{self.port}"
            self.api_started.emit(url)
            self.api_message.emit(f"API server started: {url}")
            self._server.serve_forever()
        except Exception as exc:  # noqa: BLE001
            logging.error("API server failed: %s\n%s", exc, traceback.format_exc())
            self.api_failed.emit(str(exc))
        finally:
            self.api_message.emit("API server stopped.")
            self.api_stopped.emit()

    def stop_server(self) -> None:
        """Ask Werkzeug to stop accepting requests and return from serve_forever."""
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not stop API server cleanly: %s", exc)


def parse_api_payload(request) -> Dict[str, Any]:
    """Read JSON or form fields from a Flask request without throwing on empty bodies."""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload
    if request.form:
        return dict(request.form.items())
    return {}


def resolve_api_image_path(
    request,
    payload: Dict[str, Any],
    field_name: str,
    json_key: str,
    fallback_path: str,
    temporary_directory: str,
) -> str:
    """Resolve an API image either from multipart upload, JSON path, or GUI default path."""
    uploaded_file = request.files.get(field_name)
    if uploaded_file is not None and uploaded_file.filename:
        return save_upload_to_temporary_file(uploaded_file, temporary_directory, field_name)
    return str(payload.get(json_key) or fallback_path or "").strip()


def save_upload_to_temporary_file(uploaded_file: Any, temporary_directory: str, prefix: str) -> str:
    """Save an uploaded image to a temporary file path readable by DeepFace."""
    original_suffix = Path(uploaded_file.filename or "").suffix.lower()
    if original_suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        original_suffix = ".jpg"
    safe_name = f"{prefix}_{datetime.now().strftime('%H%M%S_%f')}{original_suffix}"
    destination_path = Path(temporary_directory) / safe_name
    uploaded_file.save(str(destination_path))
    return str(destination_path)


def validate_existing_file(file_path: str, field_name: str) -> None:
    if not file_path or not Path(file_path).is_file():
        raise ValueError(f"{field_name} does not exist or is not a valid file: {file_path}")


def validate_existing_directory(directory_path: str, field_name: str) -> None:
    if not directory_path or not Path(directory_path).is_dir():
        raise ValueError(f"{field_name} does not exist or is not a valid directory: {directory_path}")


def build_api_config_from_payload(
    base_config: FaceSearchConfig,
    payload: Dict[str, Any],
    target_image_path: str,
    search_directory_path: str,
    output_parent_directory: str,
) -> FaceSearchConfig:
    """
    Build a recognition configuration for one API request.

    Every key is optional except the concrete input path needed by the endpoint. Missing values reuse
    the GUI settings captured when the API server was started.
    """
    config_values = asdict(base_config)
    config_values.update({
        "target_image_path": target_image_path,
        "search_directory_path": search_directory_path,
        "output_parent_directory": output_parent_directory,
    })

    typed_overrides = {
        "model_name": str,
        "detector_backend": str,
        "distance_metric": str,
        "normalization": str,
        "threshold": float,
        "use_pretrained_threshold": to_bool,
        "fallback_candidates": int,
        "enforce_detection": to_bool,
        "align_faces": to_bool,
        "expand_percentage": int,
        "minimum_face_confidence": float,
        "recursive_search": to_bool,
        "copy_selected_images": to_bool,
        "save_all_ranked_csv": to_bool,
        "save_summary_json": to_bool,
        "auto_open_output_directory": to_bool,
        "max_live_rows": int,
        "allow_model_fallback": to_bool,
    }
    for key, converter in typed_overrides.items():
        if key in payload and payload.get(key) is not None and payload.get(key) != "":
            try:
                config_values[key] = converter(payload.get(key))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid value for parameter {key}: {payload.get(key)}") from exc

    return FaceSearchConfig(**config_values)


def to_bool(value: Any) -> bool:
    """Parse common JSON/form boolean values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def run_api_compare(config: FaceSearchConfig, candidate_image_path: str) -> Dict[str, Any]:
    """Run a single candidate comparison and return a JSON-serializable payload."""
    started_at = datetime.now().isoformat(timespec="seconds")
    engine = FaceRecognitionEngine(config)
    target_embedding, _target_face, target_face_count = engine.get_target_embedding()
    effective_threshold, threshold_source = engine.load_threshold()
    candidate_records = engine.compare_image_to_target(target_embedding, candidate_image_path)
    selected_records, all_ranked_records, result_mode = engine.finalize_records(
        all_records=candidate_records,
        threshold=effective_threshold,
        fallback_candidates=config.fallback_candidates,
    )
    best_record = all_ranked_records[0] if all_ranked_records else None

    return {
        "ok": True,
        "endpoint": "/api/compare",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "target_face_count": target_face_count,
        "candidate_faces_compared": len(all_ranked_records),
        "result_mode": result_mode,
        "threshold": effective_threshold,
        "threshold_source": threshold_source,
        "actual_model_name": engine.actual_model_name,
        "detector_backend": config.detector_backend,
        "distance_metric": config.distance_metric,
        "best_distance": None if best_record is None else best_record.distance,
        "best_similarity_percent": None if best_record is None else best_record.similarity_percent,
        "selected_results": [asdict(record) for record in selected_records],
        "all_ranked_results": [asdict(record) for record in all_ranked_records],
    }


def run_api_directory_search(config: FaceSearchConfig, include_all_ranked: bool = False) -> Dict[str, Any]:
    """Run a synchronous directory search for API callers."""
    started_at = datetime.now().isoformat(timespec="seconds")
    engine = FaceRecognitionEngine(config)
    target_embedding, _target_face, target_face_count = engine.get_target_embedding()
    effective_threshold, threshold_source = engine.load_threshold()
    image_files = engine.iter_image_files(config.search_directory_path, config.recursive_search)
    target_absolute_path = str(Path(config.target_image_path).resolve())
    image_files = [path for path in image_files if str(Path(path).resolve()) != target_absolute_path]

    all_records: List[FaceComparisonRecord] = []
    processed_image_count = 0
    files_without_faces = 0
    failed_file_count = 0

    for image_path in image_files:
        try:
            records = engine.compare_image_to_target(target_embedding, image_path)
            processed_image_count += 1
            if records:
                all_records.extend(records)
            else:
                files_without_faces += 1
        except Exception:  # noqa: BLE001
            processed_image_count += 1
            failed_file_count += 1
            logging.warning("API search skipped image %s\n%s", image_path, traceback.format_exc())

    selected_records, all_ranked_records, result_mode = engine.finalize_records(
        all_records=all_records,
        threshold=effective_threshold,
        fallback_candidates=config.fallback_candidates,
    ) if all_records else ([], [], "no_comparable_face")

    output_directory = engine.create_output_directory(config.output_parent_directory)
    engine.write_records_csv(output_directory / "selected_results.csv", selected_records)
    if config.save_all_ranked_csv:
        engine.write_records_csv(output_directory / "all_faces_ranked.csv", all_ranked_records)
    if config.copy_selected_images:
        engine.copy_selected_images(selected_records, output_directory)

    threshold_matches_count = len([record for record in all_ranked_records if record.is_match])
    best_record = all_ranked_records[0] if all_ranked_records else None
    summary = FaceSearchSummary(
        started_at=started_at,
        finished_at=datetime.now().isoformat(timespec="seconds"),
        target_image_path=config.target_image_path,
        search_directory_path=config.search_directory_path,
        output_directory_path=str(output_directory),
        requested_model_name=config.model_name,
        actual_model_name=engine.actual_model_name,
        detector_backend=config.detector_backend,
        distance_metric=config.distance_metric,
        threshold=effective_threshold,
        threshold_source=threshold_source,
        fallback_candidates=config.fallback_candidates,
        image_files_total=len(image_files),
        image_files_processed=processed_image_count,
        faces_compared=len(all_ranked_records),
        files_without_detected_faces=files_without_faces,
        files_failed=failed_file_count,
        threshold_matches_count=threshold_matches_count,
        selected_results_count=len(selected_records),
        result_mode=result_mode,
        best_distance=None if best_record is None else best_record.distance,
        best_similarity_percent=None if best_record is None else best_record.similarity_percent,
    )
    if config.save_summary_json:
        engine.write_summary_json(output_directory / "summary.json", summary)

    response_payload = {
        "ok": True,
        "endpoint": "/api/search",
        "target_face_count": target_face_count,
        "summary": asdict(summary),
        "selected_results": [asdict(record) for record in selected_records],
    }
    if include_all_ranked:
        response_payload["all_ranked_results"] = [asdict(record) for record in all_ranked_records]
    return response_payload


class FaceSearchMainWindow(QMainWindow):
    """Main PyQt6 window with separated configuration, live monitor, final results and logs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 920)
        self.worker: Optional[FaceSearchWorker] = None
        self.api_thread: Optional[FlaskApiServerThread] = None
        self.last_output_directory: Optional[str] = None
        self.final_selected_records: List[Dict[str, Any]] = []
        self.final_all_ranked_records: List[Dict[str, Any]] = []
        self.pending_live_records: List[Dict[str, Any]] = []
        self.live_row_counter = 0

        self.live_flush_timer = QTimer(self)
        self.live_flush_timer.setInterval(180)
        self.live_flush_timer.timeout.connect(self.flush_pending_live_records)
        self.live_flush_timer.start()

        self.build_user_interface()
        self.apply_eye_friendly_no_blue_stylesheet()
        self.connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_user_interface(self) -> None:  # noqa: PLR0915
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        title_label = QLabel("High-Accuracy Local Face Search")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(title_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(False)
        root_layout.addWidget(self.tab_widget, stretch=1)

        self.setup_tab = QWidget()
        self.live_tab = QWidget()
        self.final_tab = QWidget()
        self.api_tab = QWidget()
        self.log_tab = QWidget()
        self.tab_widget.addTab(self.setup_tab, "Task and Parameters")
        self.tab_widget.addTab(self.live_tab, "Live Monitor")
        self.tab_widget.addTab(self.final_tab, "Final Results")
        self.tab_widget.addTab(self.api_tab, "POST API")
        self.tab_widget.addTab(self.log_tab, "Logs")

        self.build_setup_tab()
        self.build_live_tab()
        self.build_final_tab()
        self.build_api_tab()
        self.build_log_tab()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)

    def build_setup_tab(self) -> None:  # noqa: PLR0915
        layout = QVBoxLayout(self.setup_tab)
        layout.setSpacing(12)

        path_group = QGroupBox("Input Paths")
        path_grid = QGridLayout(path_group)
        path_grid.setHorizontalSpacing(10)
        path_grid.setVerticalSpacing(10)

        self.target_image_edit = QLineEdit()
        self.target_image_edit.setPlaceholderText("Select the target face image to search for")
        self.choose_target_button = QPushButton("Select Target Image")

        self.search_directory_edit = QLineEdit()
        self.search_directory_edit.setPlaceholderText("Select the image directory to search recursively")
        self.choose_search_directory_button = QPushButton("Select Search Directory")

        self.output_directory_edit = QLineEdit()
        self.output_directory_edit.setPlaceholderText("Defaults to the search directory; you may choose a separate output location")
        self.choose_output_directory_button = QPushButton("Select Output Directory")

        self.target_preview_label = QLabel("Target Preview")
        self.target_preview_label.setObjectName("ImagePreviewLabel")
        self.target_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_preview_label.setMinimumSize(QSize(260, 260))
        self.target_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        path_grid.addWidget(QLabel("Target image:"), 0, 0)
        path_grid.addWidget(self.target_image_edit, 0, 1)
        path_grid.addWidget(self.choose_target_button, 0, 2)
        path_grid.addWidget(QLabel("Search directory:"), 1, 0)
        path_grid.addWidget(self.search_directory_edit, 1, 1)
        path_grid.addWidget(self.choose_search_directory_button, 1, 2)
        path_grid.addWidget(QLabel("Output directory:"), 2, 0)
        path_grid.addWidget(self.output_directory_edit, 2, 1)
        path_grid.addWidget(self.choose_output_directory_button, 2, 2)
        path_grid.addWidget(self.target_preview_label, 0, 3, 4, 1)
        layout.addWidget(path_group)

        parameter_group = QGroupBox("High-Accuracy Recognition Parameters")
        parameter_grid = QGridLayout(parameter_group)
        parameter_grid.setHorizontalSpacing(12)
        parameter_grid.setVerticalSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            AUTO_HIGH_ACCURACY_MODEL,
            "Buffalo_L",
            "ArcFace",
            "Facenet512",
            "GhostFaceNet",
            "SFace",
            "Facenet",
            "VGG-Face",
            "Dlib",
            "OpenFace",
            "DeepFace",
            "DeepID",
        ])
        self.model_combo.setCurrentText(DEFAULT_MODEL_NAME)
        self.model_combo.setToolTip("By default, the app tries Buffalo_L first. If the local DeepFace build does not support it, it can fall back to ArcFace.")

        self.detector_combo = QComboBox()
        self.detector_combo.addItems([
            "retinaface",
            "mtcnn",
            "yolov8",
            "yunet",
            "mediapipe",
            "centerface",
            "ssd",
            "opencv",
            "skip",
        ])
        self.detector_combo.setCurrentText(DEFAULT_DETECTOR_BACKEND)
        self.detector_combo.setToolTip("For accuracy, retinaface is recommended. skip is suitable only for already-cropped face images.")

        self.distance_metric_combo = QComboBox()
        self.distance_metric_combo.addItems(["cosine", "euclidean_l2", "euclidean", "angular"])
        self.distance_metric_combo.setCurrentText(DEFAULT_DISTANCE_METRIC)

        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["base", "raw", "Facenet", "Facenet2018", "VGGFace", "VGGFace2", "ArcFace"])
        self.normalization_combo.setCurrentText(DEFAULT_NORMALIZATION)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 10.0000)
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(DEFAULT_MANUAL_THRESHOLD)
        self.threshold_spin.setToolTip("Manual threshold: distance <= threshold is marked as a match. When the preset threshold is enabled, this value is only a fallback.")

        self.use_pretrained_threshold_check = QCheckBox("Prefer DeepFace preset threshold")
        self.use_pretrained_threshold_check.setChecked(True)

        self.fallback_candidates_spin = QSpinBox()
        self.fallback_candidates_spin.setRange(1, 5000)
        self.fallback_candidates_spin.setValue(DEFAULT_FALLBACK_CANDIDATES)
        self.fallback_candidates_spin.setToolTip("These nearest candidates are shown only when no threshold matches are found.")

        self.minimum_confidence_spin = QDoubleSpinBox()
        self.minimum_confidence_spin.setRange(0.0, 1.0)
        self.minimum_confidence_spin.setDecimals(3)
        self.minimum_confidence_spin.setSingleStep(0.05)
        self.minimum_confidence_spin.setValue(0.0)
        self.minimum_confidence_spin.setToolTip("Faces below this detection confidence are ignored. Use 0 to disable extra filtering.")

        self.expand_percentage_spin = QSpinBox()
        self.expand_percentage_spin.setRange(0, 100)
        self.expand_percentage_spin.setValue(0)
        self.expand_percentage_spin.setToolTip("Expand the detected face area by this percentage. Default is 0.")

        parameter_grid.addWidget(QLabel("Recognition model:"), 0, 0)
        parameter_grid.addWidget(self.model_combo, 0, 1)
        parameter_grid.addWidget(QLabel("Face detector:"), 0, 2)
        parameter_grid.addWidget(self.detector_combo, 0, 3)
        parameter_grid.addWidget(QLabel("Distance metric:"), 1, 0)
        parameter_grid.addWidget(self.distance_metric_combo, 1, 1)
        parameter_grid.addWidget(QLabel("Normalization:"), 1, 2)
        parameter_grid.addWidget(self.normalization_combo, 1, 3)
        parameter_grid.addWidget(QLabel("Manual threshold:"), 2, 0)
        parameter_grid.addWidget(self.threshold_spin, 2, 1)
        parameter_grid.addWidget(self.use_pretrained_threshold_check, 2, 2, 1, 2)
        parameter_grid.addWidget(QLabel("Fallback count when no match:"), 3, 0)
        parameter_grid.addWidget(self.fallback_candidates_spin, 3, 1)
        parameter_grid.addWidget(QLabel("Minimum detection confidence:"), 3, 2)
        parameter_grid.addWidget(self.minimum_confidence_spin, 3, 3)
        parameter_grid.addWidget(QLabel("Face box expansion %:"), 4, 0)
        parameter_grid.addWidget(self.expand_percentage_spin, 4, 1)
        layout.addWidget(parameter_group)

        feature_group = QGroupBox("Feature Options")
        feature_grid = QGridLayout(feature_group)
        self.enforce_detection_check = QCheckBox("Require face detection")
        self.enforce_detection_check.setChecked(True)
        self.align_faces_check = QCheckBox("Enable face alignment")
        self.align_faces_check.setChecked(True)
        self.recursive_search_check = QCheckBox("Scan subdirectories recursively")
        self.recursive_search_check.setChecked(True)
        self.copy_selected_images_check = QCheckBox("Copy final result images")
        self.copy_selected_images_check.setChecked(True)
        self.save_all_ranked_csv_check = QCheckBox("Save full ranked CSV")
        self.save_all_ranked_csv_check.setChecked(True)
        self.save_summary_json_check = QCheckBox("Save summary.json")
        self.save_summary_json_check.setChecked(True)
        self.auto_open_output_check = QCheckBox("Open output directory when finished")
        self.auto_open_output_check.setChecked(False)
        self.allow_model_fallback_check = QCheckBox("Fall back to ArcFace if the automatic high-accuracy model is unavailable")
        self.allow_model_fallback_check.setChecked(True)

        self.max_live_rows_spin = QSpinBox()
        self.max_live_rows_spin.setRange(100, 100000)
        self.max_live_rows_spin.setValue(DEFAULT_MAX_LIVE_ROWS)
        self.max_live_rows_spin.setToolTip("Maximum rows kept in the live table. Final results are still fully saved to CSV.")

        feature_grid.addWidget(self.enforce_detection_check, 0, 0)
        feature_grid.addWidget(self.align_faces_check, 0, 1)
        feature_grid.addWidget(self.recursive_search_check, 0, 2)
        feature_grid.addWidget(self.copy_selected_images_check, 1, 0)
        feature_grid.addWidget(self.save_all_ranked_csv_check, 1, 1)
        feature_grid.addWidget(self.save_summary_json_check, 1, 2)
        feature_grid.addWidget(self.auto_open_output_check, 2, 0)
        feature_grid.addWidget(self.allow_model_fallback_check, 2, 1)
        feature_grid.addWidget(QLabel("Maximum live-table rows:"), 2, 2)
        feature_grid.addWidget(self.max_live_rows_spin, 2, 3)
        layout.addWidget(feature_group)

        self.precision_note_label = QLabel(
            "Result rule: when threshold matches exist, all matches are shown with no Top-K truncation; nearest candidates are returned only when no match exists."
            " similarity_percent is a display score converted from distance; it is not a probability."
        )
        self.precision_note_label.setObjectName("NoteLabel")
        self.precision_note_label.setWordWrap(True)
        layout.addWidget(self.precision_note_label)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Search")
        self.start_button.setObjectName("PrimaryButton")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.clear_button = QPushButton("Clear UI")
        self.open_output_directory_button = QPushButton("Open Output Directory")
        self.open_output_directory_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.open_output_directory_button)
        layout.addLayout(control_layout)
        layout.addStretch(1)

    def build_live_tab(self) -> None:
        layout = QVBoxLayout(self.live_tab)
        layout.setSpacing(12)

        progress_group = QGroupBox("Live Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.current_file_label = QLabel("Current file: -")
        self.statistics_label = QLabel("Total images 0 | Processed 0 | Faces compared 0 | Matches 0 | No face 0 | Failed 0 | Best distance - | Best similarity -")
        self.best_file_label = QLabel("Current best image: -")
        self.current_file_label.setWordWrap(True)
        self.best_file_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.current_file_label)
        progress_layout.addWidget(self.statistics_label)
        progress_layout.addWidget(self.best_file_label)
        layout.addWidget(progress_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.live_table = self.create_result_table()
        self.live_preview_label = QLabel("Live Preview")
        self.live_preview_label.setObjectName("ImagePreviewLabel")
        self.live_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_preview_label.setMinimumSize(QSize(360, 360))
        self.live_preview_label.setWordWrap(True)
        splitter.addWidget(self.live_table)
        splitter.addWidget(self.live_preview_label)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

    def build_final_tab(self) -> None:
        layout = QVBoxLayout(self.final_tab)
        layout.setSpacing(12)

        self.final_summary_label = QLabel("Final results have not been generated yet.")
        self.final_summary_label.setObjectName("NoteLabel")
        self.final_summary_label.setWordWrap(True)
        layout.addWidget(self.final_summary_label)

        top_controls = QHBoxLayout()
        self.final_view_combo = QComboBox()
        self.final_view_combo.addItems(["Final selected results", "Full ranked results"])
        self.export_selected_button = QPushButton("Export current table to CSV again")
        self.export_selected_button.setEnabled(False)
        top_controls.addWidget(QLabel("View:"))
        top_controls.addWidget(self.final_view_combo)
        top_controls.addStretch(1)
        top_controls.addWidget(self.export_selected_button)
        layout.addLayout(top_controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.final_table = self.create_result_table()
        self.final_preview_label = QLabel("Final Result Preview")
        self.final_preview_label.setObjectName("ImagePreviewLabel")
        self.final_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.final_preview_label.setMinimumSize(QSize(360, 360))
        self.final_preview_label.setWordWrap(True)
        splitter.addWidget(self.final_table)
        splitter.addWidget(self.final_preview_label)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

    def build_api_tab(self) -> None:
        """Build the optional local Flask API control panel."""
        layout = QVBoxLayout(self.api_tab)
        layout.setSpacing(12)

        server_group = QGroupBox("Local POST API Server")
        server_grid = QGridLayout(server_group)
        server_grid.setHorizontalSpacing(12)
        server_grid.setVerticalSpacing(10)

        self.api_host_edit = QLineEdit("127.0.0.1")
        self.api_host_edit.setToolTip("By default, only localhost is listened on. Use 0.0.0.0 only when LAN access is explicitly needed.")
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(8765)

        self.start_api_button = QPushButton("Start API")
        self.start_api_button.setObjectName("PrimaryButton")
        self.stop_api_button = QPushButton("Stop API")
        self.stop_api_button.setEnabled(False)
        self.api_status_label = QLabel("API status: not running")
        self.api_status_label.setObjectName("NoteLabel")
        self.api_status_label.setWordWrap(True)

        server_grid.addWidget(QLabel("Host:"), 0, 0)
        server_grid.addWidget(self.api_host_edit, 0, 1)
        server_grid.addWidget(QLabel("Port:"), 0, 2)
        server_grid.addWidget(self.api_port_spin, 0, 3)
        server_grid.addWidget(self.start_api_button, 1, 0)
        server_grid.addWidget(self.stop_api_button, 1, 1)
        server_grid.addWidget(self.api_status_label, 2, 0, 1, 4)
        layout.addWidget(server_group)

        usage_group = QGroupBox("API Usage")
        usage_layout = QVBoxLayout(usage_group)
        self.api_usage_text = QTextEdit()
        self.api_usage_text.setReadOnly(True)
        self.api_usage_text.setPlainText(
            "After startup, the following endpoints are available:\n\n"
            "1) Health check\n"
            "GET http://127.0.0.1:8765/api/health\n\n"
            "2) Single-image face comparison: POST /api/compare\n"
            "JSON example:\n"
            "{\n"
            "  \"target_image_path\": \"D:/faces/target.jpg\",\n"
            "  \"candidate_image_path\": \"D:/photos/candidate.jpg\",\n"
            "  \"model_name\": \"Automatic Highest Accuracy\",\n"
            "  \"detector_backend\": \"retinaface\",\n"
            "  \"distance_metric\": \"cosine\"\n"
            "}\n\n"
            "If a target image has already been selected in the GUI, target_image_path can be omitted.\n"
            "multipart/form-data is also supported: target_image=<file>, candidate_image=<file>.\n\n"
            "3) Directory search: POST /api/search\n"
            "JSON example:\n"
            "{\n"
            "  \"target_image_path\": \"D:/faces/target.jpg\",\n"
            "  \"search_directory_path\": \"D:/photos\",\n"
            "  \"include_all_ranked\": false\n"
            "}\n\n"
            "Note: the API reads the current GUI parameters as defaults at startup; POST fields with the same names override them.\n"
            "The API listens only on 127.0.0.1 by default. Do not expose an endpoint that can read local file paths to an untrusted network."
        )
        usage_layout.addWidget(self.api_usage_text)
        layout.addWidget(usage_group, stretch=1)

    def build_log_tab(self) -> None:
        layout = QVBoxLayout(self.log_tab)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Runtime logs are displayed here.")
        layout.addWidget(self.log_text_edit, stretch=1)

    @staticmethod
    def create_result_table() -> QTableWidget:
        table = QTableWidget(0, 12)
        table.setHorizontalHeaderLabels([
            "Rank",
            "Match",
            "Mode",
            "Distance (lower is better)",
            "Similarity % (higher is better)",
            "Model",
            "Detector",
            "Distance metric",
            "Image path",
            "Face index",
            "Detection confidence",
            "Face area",
        ])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        for column in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]:
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        table.setWordWrap(False)
        return table

    # ------------------------------------------------------------------
    # Styling and signals
    # ------------------------------------------------------------------
    def apply_eye_friendly_no_blue_stylesheet(self) -> None:
        """Apply an explicit no-blue-channel stylesheet."""
        self.setStyleSheet(
            """
            QWidget {
                background: rgb(224, 232, 0);
                color: rgb(0, 0, 0);
                font-size: 16px;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: 800;
                color: rgb(48, 80, 0);
                padding: 8px;
            }
            QLabel#NoteLabel {
                background: rgb(240, 240, 0);
                border: 2px solid rgb(120, 136, 0);
                border-radius: 8px;
                padding: 10px;
                color: rgb(48, 72, 0);
                font-weight: 600;
            }
            QLabel#ImagePreviewLabel {
                background: rgb(208, 216, 0);
                border: 3px solid rgb(96, 112, 0);
                border-radius: 10px;
                color: rgb(48, 64, 0);
                font-weight: 700;
            }
            QTabWidget::pane {
                border: 2px solid rgb(96, 112, 0);
                border-radius: 8px;
                background: rgb(232, 236, 0);
            }
            QTabBar::tab {
                background: rgb(200, 216, 0);
                color: rgb(0, 0, 0);
                border: 2px solid rgb(96, 112, 0);
                border-bottom: 0px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 18px;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                background: rgb(240, 240, 0);
            }
            QGroupBox {
                border: 2px solid rgb(96, 120, 0);
                border-radius: 9px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 800;
                background: rgb(232, 236, 0);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0px 8px 0px 8px;
                background: rgb(232, 236, 0);
                color: rgb(40, 72, 0);
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
                background: rgb(248, 248, 0);
                color: rgb(0, 0, 0);
                border: 2px solid rgb(120, 136, 0);
                border-radius: 6px;
                padding: 7px;
                selection-background-color: rgb(128, 144, 0);
                selection-color: rgb(0, 0, 0);
            }
            QComboBox::drop-down {
                border-left: 2px solid rgb(120, 136, 0);
                background: rgb(224, 232, 0);
                width: 28px;
            }
            QPushButton {
                background: rgb(176, 192, 0);
                color: rgb(0, 0, 0);
                border: 2px solid rgb(88, 104, 0);
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 800;
            }
            QPushButton:hover {
                background: rgb(192, 208, 0);
            }
            QPushButton:pressed {
                background: rgb(144, 160, 0);
            }
            QPushButton:disabled {
                background: rgb(184, 184, 0);
                color: rgb(96, 96, 0);
                border-color: rgb(144, 144, 0);
            }
            QPushButton#PrimaryButton {
                background: rgb(160, 208, 0);
                border: 3px solid rgb(64, 112, 0);
            }
            QCheckBox {
                spacing: 8px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid rgb(96, 112, 0);
                background: rgb(248, 248, 0);
            }
            QCheckBox::indicator:checked {
                background: rgb(128, 176, 0);
            }
            QProgressBar {
                border: 2px solid rgb(96, 112, 0);
                border-radius: 7px;
                background: rgb(248, 248, 0);
                color: rgb(0, 0, 0);
                text-align: center;
                height: 28px;
                font-weight: 800;
            }
            QProgressBar::chunk {
                background: rgb(112, 176, 0);
                border-radius: 5px;
            }
            QHeaderView::section {
                background: rgb(192, 208, 0);
                color: rgb(0, 0, 0);
                border: 1px solid rgb(96, 112, 0);
                padding: 8px;
                font-weight: 800;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background: rgb(128, 144, 0);
                color: rgb(0, 0, 0);
            }
            QScrollBar:vertical {
                background: rgb(224, 232, 0);
                width: 16px;
            }
            QScrollBar::handle:vertical {
                background: rgb(144, 160, 0);
                min-height: 30px;
                border-radius: 6px;
            }
            QSplitter::handle {
                background: rgb(120, 136, 0);
            }
            QMessageBox {
                background: rgb(224, 232, 0);
            }
            """
        )

    def connect_signals(self) -> None:
        self.choose_target_button.clicked.connect(self.choose_target_image)
        self.choose_search_directory_button.clicked.connect(self.choose_search_directory)
        self.choose_output_directory_button.clicked.connect(self.choose_output_directory)
        self.target_image_edit.textChanged.connect(self.refresh_target_preview)
        self.search_directory_edit.textChanged.connect(self.fill_default_output_directory)
        self.start_button.clicked.connect(self.start_search)
        self.stop_button.clicked.connect(self.stop_search)
        self.clear_button.clicked.connect(self.clear_interface)
        self.open_output_directory_button.clicked.connect(self.open_output_directory)
        self.export_selected_button.clicked.connect(self.export_current_final_table)
        self.final_view_combo.currentIndexChanged.connect(self.refresh_final_table)
        self.live_table.itemSelectionChanged.connect(lambda: self.update_preview_from_selected_row(self.live_table, self.live_preview_label))
        self.final_table.itemSelectionChanged.connect(lambda: self.update_preview_from_selected_row(self.final_table, self.final_preview_label))
        self.live_table.itemDoubleClicked.connect(self.open_table_item_path)
        self.final_table.itemDoubleClicked.connect(self.open_table_item_path)
        self.start_api_button.clicked.connect(self.start_api_server)
        self.stop_api_button.clicked.connect(self.stop_api_server)

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------
    def choose_target_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target Face Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All Files (*)",
        )
        if file_path:
            self.target_image_edit.setText(file_path)

    def choose_search_directory(self) -> None:
        directory_path = QFileDialog.getExistingDirectory(self, "Select Directory to Search")
        if directory_path:
            self.search_directory_edit.setText(directory_path)
            if not self.output_directory_edit.text().strip():
                self.output_directory_edit.setText(directory_path)

    def choose_output_directory(self) -> None:
        directory_path = QFileDialog.getExistingDirectory(self, "Select Result Output Directory")
        if directory_path:
            self.output_directory_edit.setText(directory_path)

    def fill_default_output_directory(self) -> None:
        if not self.output_directory_edit.text().strip():
            search_directory = self.search_directory_edit.text().strip()
            if search_directory:
                self.output_directory_edit.setText(search_directory)

    def refresh_target_preview(self) -> None:
        image_path = self.target_image_edit.text().strip()
        if not image_path or not Path(image_path).exists():
            self.target_preview_label.setText("Target Preview")
            self.target_preview_label.setPixmap(QPixmap())
            return

        pixmap = load_no_blue_pixmap(image_path, self.target_preview_label.size())
        if pixmap is None:
            self.target_preview_label.setText("Preview unavailable")
            self.target_preview_label.setPixmap(QPixmap())
            return

        self.target_preview_label.setText("")
        self.target_preview_label.setPixmap(pixmap)

    def validate_inputs(self) -> Optional[FaceSearchConfig]:
        target_image_path = self.target_image_edit.text().strip()
        search_directory_path = self.search_directory_edit.text().strip()
        output_directory_path = self.output_directory_edit.text().strip() or search_directory_path

        if not target_image_path or not Path(target_image_path).is_file():
            self.show_warning("Please select a valid target face image first.")
            return None
        if not search_directory_path or not Path(search_directory_path).is_dir():
            self.show_warning("Please select a valid search directory first.")
            return None
        if not output_directory_path or not Path(output_directory_path).exists():
            try:
                Path(output_directory_path).mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                self.show_warning(f"Output directory is unavailable: {exc}")
                return None

        return FaceSearchConfig(
            target_image_path=target_image_path,
            search_directory_path=search_directory_path,
            output_parent_directory=output_directory_path,
            model_name=self.model_combo.currentText(),
            detector_backend=self.detector_combo.currentText(),
            distance_metric=self.distance_metric_combo.currentText(),
            threshold=float(self.threshold_spin.value()),
            use_pretrained_threshold=self.use_pretrained_threshold_check.isChecked(),
            fallback_candidates=int(self.fallback_candidates_spin.value()),
            enforce_detection=self.enforce_detection_check.isChecked(),
            align_faces=self.align_faces_check.isChecked(),
            expand_percentage=int(self.expand_percentage_spin.value()),
            normalization=self.normalization_combo.currentText(),
            minimum_face_confidence=float(self.minimum_confidence_spin.value()),
            recursive_search=self.recursive_search_check.isChecked(),
            copy_selected_images=self.copy_selected_images_check.isChecked(),
            save_all_ranked_csv=self.save_all_ranked_csv_check.isChecked(),
            save_summary_json=self.save_summary_json_check.isChecked(),
            auto_open_output_directory=self.auto_open_output_check.isChecked(),
            max_live_rows=int(self.max_live_rows_spin.value()),
            allow_model_fallback=self.allow_model_fallback_check.isChecked(),
        )

    def start_search(self) -> None:
        config = self.validate_inputs()
        if config is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self.show_warning("A search task is still running.")
            return

        self.reset_runtime_views()
        self.append_log("Search task started.")
        self.append_log("The interface will refresh in real time. Final results will be re-ranked by the threshold rule when the task finishes.")
        self.tab_widget.setCurrentWidget(self.live_tab)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_output_directory_button.setEnabled(False)
        self.export_selected_button.setEnabled(False)

        self.worker = FaceSearchWorker(config)
        self.worker.progress_changed.connect(self.handle_progress_changed)
        self.worker.statistics_changed.connect(self.handle_statistics_changed)
        self.worker.live_records_ready.connect(self.queue_live_records)
        self.worker.message_ready.connect(self.append_log)
        self.worker.task_succeeded.connect(self.handle_task_succeeded)
        self.worker.task_failed.connect(self.handle_task_failed)
        self.worker.finished.connect(self.handle_worker_finished)
        self.worker.start()

    def stop_search(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.append_log("Stop requested. The app will organize completed comparisons instead of forcefully interrupting the process.")

    def clear_interface(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.show_warning("The search is running, so the UI cannot be cleared. Please stop it first.")
            return
        self.reset_runtime_views()
        self.log_text_edit.clear()
        self.append_log("The UI has been cleared.")

    def open_output_directory(self) -> None:
        if not self.last_output_directory or not Path(self.last_output_directory).exists():
            self.show_warning("There is no output directory to open yet.")
            return
        open_path_in_os(self.last_output_directory)

    def export_current_final_table(self) -> None:
        records = self.get_records_for_current_final_view()
        if not records:
            self.show_warning("There are no final results to export.")
            return

        default_path = str(Path(self.last_output_directory or str(APP_DIR)) / "current_final_table.csv")
        csv_path, _ = QFileDialog.getSaveFileName(self, "Export Current Table CSV", default_path, "CSV Files (*.csv)")
        if not csv_path:
            return

        try:
            csv_records = [dict_to_record(record) for record in records]
            FaceRecognitionEngine.write_records_csv(Path(csv_path), csv_records)
            self.append_log(f"Current table exported: {csv_path}")
        except Exception as exc:  # noqa: BLE001
            self.show_warning(f"Export failed: {exc}")

    # ------------------------------------------------------------------
    # API server actions
    # ------------------------------------------------------------------
    def build_config_snapshot_for_api(self) -> FaceSearchConfig:
        """
        Capture current GUI parameters for the optional API server.

        Unlike validate_inputs(), this method does not require paths to exist. API callers may provide
        target_image_path, candidate_image_path or search_directory_path in each POST request.
        """
        search_directory_path = self.search_directory_edit.text().strip() or str(APP_DIR)
        output_directory_path = self.output_directory_edit.text().strip() or search_directory_path or str(APP_DIR)
        target_image_path = self.target_image_edit.text().strip()

        return FaceSearchConfig(
            target_image_path=target_image_path,
            search_directory_path=search_directory_path,
            output_parent_directory=output_directory_path,
            model_name=self.model_combo.currentText(),
            detector_backend=self.detector_combo.currentText(),
            distance_metric=self.distance_metric_combo.currentText(),
            threshold=float(self.threshold_spin.value()),
            use_pretrained_threshold=self.use_pretrained_threshold_check.isChecked(),
            fallback_candidates=int(self.fallback_candidates_spin.value()),
            enforce_detection=self.enforce_detection_check.isChecked(),
            align_faces=self.align_faces_check.isChecked(),
            expand_percentage=int(self.expand_percentage_spin.value()),
            normalization=self.normalization_combo.currentText(),
            minimum_face_confidence=float(self.minimum_confidence_spin.value()),
            recursive_search=self.recursive_search_check.isChecked(),
            copy_selected_images=self.copy_selected_images_check.isChecked(),
            save_all_ranked_csv=self.save_all_ranked_csv_check.isChecked(),
            save_summary_json=self.save_summary_json_check.isChecked(),
            auto_open_output_directory=False,
            max_live_rows=int(self.max_live_rows_spin.value()),
            allow_model_fallback=self.allow_model_fallback_check.isChecked(),
        )

    def start_api_server(self) -> None:
        """Start the local Flask API in a background QThread."""
        if self.api_thread is not None and self.api_thread.isRunning():
            self.show_warning("The API is already running.")
            return

        host = self.api_host_edit.text().strip() or "127.0.0.1"
        port = int(self.api_port_spin.value())
        if host == "0.0.0.0":
            reply = QMessageBox.question(
                self,
                "Confirm listening on all network interfaces",
                "0.0.0.0 may allow other devices on the LAN to access this API.\n"
                "This API can read local file paths provided in requests. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        config_snapshot = self.build_config_snapshot_for_api()
        self.api_thread = FlaskApiServerThread(host=host, port=port, default_config=config_snapshot)
        self.api_thread.api_started.connect(self.handle_api_started)
        self.api_thread.api_stopped.connect(self.handle_api_stopped)
        self.api_thread.api_failed.connect(self.handle_api_failed)
        self.api_thread.api_message.connect(self.append_log)
        self.api_thread.start()

        self.start_api_button.setEnabled(False)
        self.stop_api_button.setEnabled(True)
        self.api_status_label.setText("API status: starting...")
        self.append_log("Starting the local Flask API server.")

    def stop_api_server(self) -> None:
        """Stop the local Flask API server."""
        if self.api_thread is not None and self.api_thread.isRunning():
            self.api_thread.stop_server()
            self.stop_api_button.setEnabled(False)
            self.api_status_label.setText("API status: stopping...")
            self.append_log("API server stop requested.")
        else:
            self.api_status_label.setText("API status: not running")

    def handle_api_started(self, base_url: str) -> None:
        self.api_status_label.setText(
            f"API status: running | {base_url}\n"
            f"POST compare endpoint: {base_url}/api/compare\n"
            f"POST directory-search endpoint: {base_url}/api/search"
        )
        self.append_log(f"API started: {base_url}")

    def handle_api_stopped(self) -> None:
        self.start_api_button.setEnabled(True)
        self.stop_api_button.setEnabled(False)
        self.api_status_label.setText("API status: not running")

    def handle_api_failed(self, error_message: str) -> None:
        self.start_api_button.setEnabled(True)
        self.stop_api_button.setEnabled(False)
        self.api_status_label.setText(f"API status: startup failed | {error_message}")
        self.show_warning(f"API startup failed:\n{error_message}")

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------
    def handle_progress_changed(self, processed: int, total: int, image_path: str) -> None:
        percentage = 0 if total <= 0 else int(processed * 100 / total)
        self.progress_bar.setValue(percentage)
        self.current_file_label.setText(f"Current file: {image_path}")

    def handle_statistics_changed(self, statistics: Dict[str, Any]) -> None:
        best_distance = statistics.get("best_distance")
        best_similarity = statistics.get("best_similarity_percent")
        best_distance_text = "-" if best_distance is None else f"{float(best_distance):.8f}"
        best_similarity_text = "-" if best_similarity is None else f"{float(best_similarity):.2f}%"
        self.statistics_label.setText(
            f"Total images {statistics.get('total_images', 0)} | "
            f"Processed {statistics.get('processed_images', 0)} | "
            f"Faces compared {statistics.get('faces_compared', 0)} | "
            f"Matches {statistics.get('matches_count', 0)} | "
            f"No face {statistics.get('files_without_faces', 0)} | "
            f"Failed {statistics.get('failed_files', 0)} | "
            f"Best distance {best_distance_text} | Best similarity {best_similarity_text}"
        )
        best_image_path = statistics.get("best_image_path")
        self.best_file_label.setText(f"Current best image: {best_image_path or '-'}")

    def queue_live_records(self, records: List[Dict[str, Any]]) -> None:
        self.pending_live_records.extend(records)

    def flush_pending_live_records(self) -> None:
        if not self.pending_live_records:
            return

        batch = self.pending_live_records[:120]
        self.pending_live_records = self.pending_live_records[120:]

        self.live_table.setUpdatesEnabled(False)
        for record in batch:
            self.live_row_counter += 1
            live_record = dict(record)
            live_record["rank"] = self.live_row_counter
            self.add_record_to_table(self.live_table, live_record)
        self.trim_live_table_if_needed()
        self.live_table.setUpdatesEnabled(True)

    def handle_task_succeeded(self, summary: Dict[str, Any], selected_records: List[Dict[str, Any]], all_ranked_records: List[Dict[str, Any]]) -> None:
        self.flush_pending_live_records()
        self.final_selected_records = selected_records
        self.final_all_ranked_records = all_ranked_records
        self.last_output_directory = summary.get("output_directory_path")

        self.open_output_directory_button.setEnabled(bool(self.last_output_directory))
        self.export_selected_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.refresh_final_table()
        self.update_final_summary(summary)

        self.append_log(f"Search completed. Output directory: {self.last_output_directory}")
        self.append_log("selected_results.csv has been written. If enabled, all_faces_ranked.csv and summary.json were also written.")

        if summary.get("result_mode") == "threshold_match_all":
            self.append_log("Final rule: threshold matches exist, so all threshold matches were returned with no Top-K truncation.")
        else:
            self.append_log("Final rule: no threshold matches were found, so the nearest candidates were returned as a fallback.")

        self.tab_widget.setCurrentWidget(self.final_tab)

        if self.worker is not None and self.worker.config.auto_open_output_directory and self.last_output_directory:
            open_path_in_os(self.last_output_directory)

    def handle_task_failed(self, error_message: str) -> None:
        self.append_log(f"Task failed: {error_message}")
        self.show_warning(f"Task failed:\n{error_message}\n\nDetailed log: {LOG_FILE}")

    def handle_worker_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    # ------------------------------------------------------------------
    # Table, previews and summaries
    # ------------------------------------------------------------------
    def reset_runtime_views(self) -> None:
        self.pending_live_records.clear()
        self.live_row_counter = 0
        self.live_table.setRowCount(0)
        self.final_table.setRowCount(0)
        self.final_selected_records = []
        self.final_all_ranked_records = []
        self.final_summary_label.setText("Final results have not been generated yet.")
        self.progress_bar.setValue(0)
        self.current_file_label.setText("Current file: -")
        self.statistics_label.setText("Total images 0 | Processed 0 | Faces compared 0 | Matches 0 | No face 0 | Failed 0 | Best distance - | Best similarity -")
        self.best_file_label.setText("Current best image: -")
        self.live_preview_label.setText("Live Preview")
        self.live_preview_label.setPixmap(QPixmap())
        self.final_preview_label.setText("Final Result Preview")
        self.final_preview_label.setPixmap(QPixmap())

    def trim_live_table_if_needed(self) -> None:
        maximum_rows = int(self.max_live_rows_spin.value())
        while self.live_table.rowCount() > maximum_rows:
            self.live_table.removeRow(0)

    def add_record_to_table(self, table: QTableWidget, record: Dict[str, Any]) -> None:
        row = table.rowCount()
        table.insertRow(row)

        is_match = bool(record.get("is_match"))
        values = [
            str(record.get("rank", row + 1)),
            "Yes" if is_match else "No",
            translate_result_mode(str(record.get("result_mode", ""))),
            format_float(record.get("distance"), 8),
            f"{format_float(record.get('similarity_percent'), 2)}%",
            str(record.get("model_name", "")),
            str(record.get("detector_backend", "")),
            str(record.get("distance_metric", "")),
            str(record.get("image_path", "")),
            str(record.get("face_index", "")),
            "" if record.get("face_confidence") is None else format_float(record.get("face_confidence"), 6),
            str(record.get("facial_area_json", "{}")),
        ]

        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setForeground(COLOR_TEXT)
            item.setBackground(COLOR_MATCH if is_match else COLOR_CANDIDATE)
            if column in {0, 1, 3, 4, 9, 10}:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, column, item)

    def refresh_final_table(self) -> None:
        records = self.get_records_for_current_final_view()
        self.final_table.setUpdatesEnabled(False)
        self.final_table.setRowCount(0)
        for record in records:
            self.add_record_to_table(self.final_table, record)
        self.final_table.setUpdatesEnabled(True)

    def get_records_for_current_final_view(self) -> List[Dict[str, Any]]:
        if self.final_view_combo.currentText() == "Full ranked results":
            return self.final_all_ranked_records
        return self.final_selected_records

    def update_final_summary(self, summary: Dict[str, Any]) -> None:
        result_mode = translate_result_mode(str(summary.get("result_mode", "")))
        best_distance = summary.get("best_distance")
        best_similarity = summary.get("best_similarity_percent")
        self.final_summary_label.setText(
            f"Final mode: {result_mode} | "
            f"Threshold matches: {summary.get('threshold_matches_count', 0)} | "
            f"Current final selection: {summary.get('selected_results_count', 0)} | "
            f"All comparable faces: {summary.get('faces_compared', 0)} | "
            f"Actual model: {summary.get('actual_model_name', '-')} | "
            f"Threshold: {format_float(summary.get('threshold'), 8)} ({summary.get('threshold_source', '-')}) | "
            f"Best distance: {format_float(best_distance, 8)} | "
            f"Best similarity: {format_float(best_similarity, 2)}% | "
            f"Output directory: {summary.get('output_directory_path', '-')}"
        )

    def update_preview_from_selected_row(self, table: QTableWidget, preview_label: QLabel) -> None:
        selected_items = table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        path_item = table.item(row, 8)
        if path_item is None:
            return
        image_path = path_item.text()
        if not image_path or not Path(image_path).exists():
            preview_label.setText("Image does not exist")
            preview_label.setPixmap(QPixmap())
            return

        pixmap = load_no_blue_pixmap(image_path, preview_label.size())
        if pixmap is None:
            preview_label.setText("Preview unavailable")
            preview_label.setPixmap(QPixmap())
            return

        preview_label.setText("")
        preview_label.setPixmap(pixmap)

    def open_table_item_path(self, item: QTableWidgetItem) -> None:
        table = item.tableWidget()
        if table is None:
            return
        path_item = table.item(item.row(), 8)
        if path_item is None:
            return
        image_path = path_item.text()
        if image_path and Path(image_path).exists():
            open_path_in_os(image_path)

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def show_warning(self, message: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Notice")
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def resizeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.refresh_target_preview()
        self.update_preview_from_selected_row(self.live_table, self.live_preview_label)
        self.update_preview_from_selected_row(self.final_table, self.final_preview_label)

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "A search is still running. Request stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.requestInterruption()
                self.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
                return
        else:
            event.accept()

        if event.isAccepted() and self.api_thread is not None and self.api_thread.isRunning():
            self.api_thread.stop_server()
            self.api_thread.wait(3000)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def sanitize_filename(value: str) -> str:
    safe_characters = []
    for character in value:
        if character.isalnum() or character in {"-", "_", "."}:
            safe_characters.append(character)
        else:
            safe_characters.append("_")
    return "".join(safe_characters)[:120] or "image"


def format_float(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:  # noqa: BLE001
        return "-"


def translate_result_mode(mode: str) -> str:
    translations = {
        "live_candidate": "Live candidate",
        "threshold_match": "Match",
        "threshold_match_all": "All threshold matches",
        "no_match_nearest_fallback": "No match, nearest fallback",
    }
    return translations.get(mode, mode)


def dict_to_record(record: Dict[str, Any]) -> FaceComparisonRecord:
    return FaceComparisonRecord(
        rank=int(record.get("rank", 0) or 0),
        image_path=str(record.get("image_path", "")),
        face_index=int(record.get("face_index", 0) or 0),
        distance=float(record.get("distance", float("inf"))),
        similarity_score=float(record.get("similarity_score", 0.0) or 0.0),
        similarity_percent=float(record.get("similarity_percent", 0.0) or 0.0),
        is_match=bool(record.get("is_match", False)),
        threshold=float(record.get("threshold", DEFAULT_MANUAL_THRESHOLD) or DEFAULT_MANUAL_THRESHOLD),
        result_mode=str(record.get("result_mode", "")),
        model_name=str(record.get("model_name", "")),
        detector_backend=str(record.get("detector_backend", "")),
        distance_metric=str(record.get("distance_metric", "")),
        face_confidence=None if record.get("face_confidence") is None else float(record.get("face_confidence")),
        facial_area_json=str(record.get("facial_area_json", "{}")),
        error_message=str(record.get("error_message", "")),
    )


def load_no_blue_pixmap(image_path: str, target_size: QSize) -> Optional[QPixmap]:
    """
    Load an image for display and set the displayed blue channel to zero.

    The image on disk is not modified. OpenCV reads BGR, so channel index 0 is the displayed blue
    channel before conversion to RGB.
    """
    if cv2 is None:
        return None

    try:
        image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None

        image_bgr[:, :, 0] = 0
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        return pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    except Exception:  # noqa: BLE001
        logging.warning("Could not load preview image %s\n%s", image_path, traceback.format_exc())
        return None


def open_path_in_os(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not open path %s: %s", path, exc)


def install_exception_hook(parent_getter=None) -> None:
    """Show uncaught exceptions in a dialog and write the full traceback to a log file."""

    def exception_hook(exception_type, exception_value, exception_traceback):
        text = "".join(traceback.format_exception(exception_type, exception_value, exception_traceback))
        logging.critical("Uncaught exception:\n%s", text)
        try:
            parent = parent_getter() if parent_getter else None
            QMessageBox.critical(parent, "Application Error", f"The application encountered an error, and a log was written to:\n{LOG_FILE}\n\n{exception_value}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = exception_hook


def apply_no_blue_palette(app: QApplication) -> None:
    """Set a no-blue RGB palette before the window is created."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, COLOR_WINDOW)
    palette.setColor(QPalette.ColorRole.WindowText, COLOR_TEXT)
    palette.setColor(QPalette.ColorRole.Base, COLOR_FIELD)
    palette.setColor(QPalette.ColorRole.AlternateBase, COLOR_PANEL)
    palette.setColor(QPalette.ColorRole.ToolTipBase, COLOR_FIELD)
    palette.setColor(QPalette.ColorRole.ToolTipText, COLOR_TEXT)
    palette.setColor(QPalette.ColorRole.Text, COLOR_TEXT)
    palette.setColor(QPalette.ColorRole.Button, COLOR_BUTTON)
    palette.setColor(QPalette.ColorRole.ButtonText, COLOR_TEXT)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 0))
    palette.setColor(QPalette.ColorRole.Highlight, COLOR_HIGHLIGHT)
    palette.setColor(QPalette.ColorRole.HighlightedText, COLOR_TEXT)
    palette.setColor(QPalette.ColorRole.Link, QColor(96, 128, 0))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(80, 96, 0))
    app.setPalette(palette)


def check_runtime_dependencies() -> Optional[str]:
    """Return a human-readable dependency error, or None when all required modules import."""
    missing_dependencies: List[str] = []
    try:
        import PyQt6  # noqa: F401
    except Exception:
        missing_dependencies.append("PyQt6")
    try:
        import deepface  # noqa: F401
    except Exception:
        missing_dependencies.append("deepface")
    try:
        import cv2 as _cv2  # noqa: F401
    except Exception:
        missing_dependencies.append("opencv-python")
    try:
        import numpy  # noqa: F401
    except Exception:
        missing_dependencies.append("numpy")

    if missing_dependencies:
        return (
            "Missing dependencies: " + ", ".join(missing_dependencies) +
            "\n\nPlease run first:\npip install PyQt6 deepface tensorflow opencv-python numpy flask"
        )
    return None


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 12))
    apply_no_blue_palette(app)
    window_reference: Dict[str, Optional[FaceSearchMainWindow]] = {"window": None}
    install_exception_hook(lambda: window_reference.get("window"))
    dependency_error = check_runtime_dependencies()
    if dependency_error:
        QMessageBox.critical(None, "Missing Dependencies", dependency_error)
        return 1
    window = FaceSearchMainWindow()
    window_reference["window"] = window
    window.show()
    return app.exec()
if __name__ == "__main__":
    raise SystemExit(main())
