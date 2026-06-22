"""
AGPL-3.0 Licensed Single-File Flask Face Search Service.
This script provides a complete web and API service for 1:N face search. It uses
dlib / face_recognition to extract 128-dimensional face embeddings, stores the
startup-built face gallery in an exact FAISS IndexFlatIP index, and returns
ranked cosine-similarity matches for each uploaded query image.
The gallery database is read only at service startup from the face_database
folder located beside this file. Web users and API clients can upload query
images for matching, but they cannot add, modify, rebuild, or delete database
images through this service.
The script prints runtime diagnostics at startup, including dlib version, CUDA
compile status, visible CUDA device count, selected face detector, encoding
model, FAISS CPU thread count, and database folder path. If dlib can use CUDA,
the service selects the CNN detector for higher detection accuracy; otherwise it
selects the CPU-friendly HOG detector.
Run: python dlib_faiss_flask_1n_single_file.py
Open: https://localhost:5000
License: GNU Affero General Public License v3.0 only (AGPL-3.0-only).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dlib
import faiss
import face_recognition
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

def get_dlib_cuda_device_count() -> int:
    """Return the number of visible CUDA devices without crashing on CPU-only dlib builds."""
    try:
        return int(dlib.cuda.get_num_devices()) if bool(getattr(dlib, "DLIB_USE_CUDA", False)) else 0
    except Exception as cuda_error:
        print(f"[face-search] CUDA device check failed: {cuda_error}", flush=True)
        return 0


DLIB_VERSION = getattr(dlib, "__version__", "unknown")                       # Installed dlib version.
IS_DLIB_COMPILED_WITH_CUDA = bool(getattr(dlib, "DLIB_USE_CUDA", False))      # Whether this dlib package was compiled with CUDA.
DLIB_CUDA_DEVICE_COUNT = get_dlib_cuda_device_count()                         # Visible CUDA device count reported by dlib.
IS_CUDA_RUNTIME_AVAILABLE = IS_DLIB_COMPILED_WITH_CUDA and DLIB_CUDA_DEVICE_COUNT > 0  # True only when CNN can realistically use GPU.
APP_FOLDER = Path(__file__).resolve().parent                                  # Folder that contains this single Flask file.
DATABASE_FOLDER = APP_FOLDER / "face_database"                                # Startup-only gallery folder.
UPLOAD_FOLDER = APP_FOLDER / "runtime_uploads"                                # Temporary folder for query images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}              # Supported image extensions.
EMBEDDING_DIMENSION = 128                                                     # dlib face embedding dimension.
DEFAULT_TOP_K = 0                                                             # 0 means return all indexed faces.
DEFAULT_MATCH_THRESHOLD = 0.60                                                # Default cosine-similarity match threshold.
FACE_DETECTION_MODEL = "cnn" if IS_CUDA_RUNTIME_AVAILABLE else "hog"          # Use high-accuracy CNN only when CUDA is available; otherwise use CPU-friendly HOG.
FACE_ENCODING_MODEL = "large"                                                 # Use dlib's large landmark model for better alignment quality.
FACE_DETECTION_UPSAMPLE_TIMES = 1 if IS_CUDA_RUNTIME_AVAILABLE else 1          # Extra upsample improves small-face recall; keep modest for CPU usability.
FACE_ENCODING_NUM_JITTERS = 2 if IS_CUDA_RUNTIME_AVAILABLE else 1              # More jitters improves stability but increases encoding time.
MAX_UPLOAD_SIZE_MB = 64                                                        # Maximum query image upload size.
FAISS_CPU_THREAD_COUNT = max(1, os.cpu_count() or 1)                           # Number of FAISS CPU threads.

DATABASE_FOLDER.mkdir(parents=True, exist_ok=True)                             # Ensure gallery folder exists.
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)                               # Ensure query temp folder exists.
faiss.omp_set_num_threads(FAISS_CPU_THREAD_COUNT)                              # Let FAISS use available CPU threads.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024            # Enforce upload size limit.


def print_runtime_configuration() -> None:
    """Print dlib, CUDA, model, path, and FAISS settings for deployment diagnostics."""
    print(f"[face-search] dlib version: {DLIB_VERSION}", flush=True)
    print(f"[face-search] dlib compiled with CUDA: {IS_DLIB_COMPILED_WITH_CUDA}", flush=True)
    print(f"[face-search] CUDA device count: {DLIB_CUDA_DEVICE_COUNT}", flush=True)
    print(f"[face-search] CUDA runtime available: {IS_CUDA_RUNTIME_AVAILABLE}", flush=True)
    print(f"[face-search] selected detection model: {FACE_DETECTION_MODEL}", flush=True)
    print(f"[face-search] selected encoding model: {FACE_ENCODING_MODEL}", flush=True)
    print(f"[face-search] upsample times: {FACE_DETECTION_UPSAMPLE_TIMES}", flush=True)
    print(f"[face-search] num jitters: {FACE_ENCODING_NUM_JITTERS}", flush=True)
    print(f"[face-search] FAISS CPU threads: {FAISS_CPU_THREAD_COUNT}", flush=True)
    print(f"[face-search] database folder: {DATABASE_FOLDER}", flush=True)


@dataclass(frozen=True)
class IndexBuildReport:
    """Summary of one startup database scan and FAISS index build."""

    database_folder: str
    scanned_image_count: int
    indexed_face_count: int
    failed_image_count: int
    skipped_image_count: int
    index_type: str
    metric: str
    faiss_thread_count: int


class FaceSearchEngine:
    """Exact cosine-similarity face search engine using dlib embeddings and FAISS."""

    def __init__(self) -> None:
        """Create locks and empty in-memory FAISS state."""
        self._state_lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._face_metadata: List[Dict[str, Any]] = []
        self._latest_report = IndexBuildReport(
            database_folder=str(DATABASE_FOLDER),
            scanned_image_count=0,
            indexed_face_count=0,
            failed_image_count=0,
            skipped_image_count=0,
            index_type="faiss.IndexFlatIP exact search",
            metric="cosine_similarity over L2-normalized dlib embeddings",
            faiss_thread_count=FAISS_CPU_THREAD_COUNT,
        )

    @staticmethod
    def _normalize_embedding(face_embedding: np.ndarray) -> np.ndarray:
        """L2-normalize one 128D dlib embedding before inserting/searching FAISS."""
        embedding = np.asarray(face_embedding, dtype=np.float64).reshape(1, EMBEDDING_DIMENSION)
        embedding_norm = float(np.linalg.norm(embedding, ord=2))
        if not np.isfinite(embedding_norm) or embedding_norm <= 0.0:
            raise ValueError("Invalid face embedding norm")
        normalized_embedding = embedding / embedding_norm
        return np.ascontiguousarray(normalized_embedding.astype(np.float32, copy=False))

    @staticmethod
    def _face_location_to_box(face_location: Tuple[int, int, int, int]) -> Dict[str, int]:
        """Convert face_recognition's location tuple into a JSON-friendly box."""
        top, right, bottom, left = face_location
        return {
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "left": int(left),
            "width": int(max(0, right - left)),
            "height": int(max(0, bottom - top)),
        }

    @staticmethod
    def _face_area(face_location: Tuple[int, int, int, int]) -> int:
        """Return detected face area so the largest query face can be selected."""
        top, right, bottom, left = face_location
        return max(0, right - left) * max(0, bottom - top)

    @staticmethod
    def _list_database_images(database_folder: Path) -> List[Path]:
        """Find all supported image files under the database folder."""
        image_paths: List[Path] = []
        if not database_folder.is_dir():
            return image_paths
        for current_folder, _, file_names in os.walk(database_folder):
            for file_name in sorted(file_names):
                image_path = Path(current_folder) / file_name
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(image_path)
        return sorted(image_paths, key=lambda path: path.as_posix())

    def _extract_face_records_from_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detect and encode every face in one gallery image."""
        image_array = face_recognition.load_image_file(str(image_path))
        detect_started_at = time.perf_counter()
        face_locations = face_recognition.face_locations(
            image_array,
            number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE_TIMES,
            model=FACE_DETECTION_MODEL,
        )
        detect_seconds = time.perf_counter() - detect_started_at
        print(f"[face-search] detected {len(face_locations)} face(s) in {detect_seconds:.2f}s", flush=True)
        if not face_locations:
            return []
        encode_started_at = time.perf_counter()
        face_encodings = face_recognition.face_encodings(
            image_array,
            known_face_locations=face_locations,
            num_jitters=FACE_ENCODING_NUM_JITTERS,
            model=FACE_ENCODING_MODEL,
        )
        encode_seconds = time.perf_counter() - encode_started_at
        print(f"[face-search] encoded {len(face_encodings)} face(s) in {encode_seconds:.2f}s", flush=True)
        face_records: List[Dict[str, Any]] = []
        for face_index, face_embedding in enumerate(face_encodings):
            face_records.append(
                {
                    "face_index": face_index,
                    "face_box": self._face_location_to_box(face_locations[face_index]),
                    "normalized_embedding": self._normalize_embedding(face_embedding),
                }
            )
        return face_records

    def rebuild_index(self, database_folder: Path = DATABASE_FOLDER) -> IndexBuildReport:
        """Re-scan gallery images, encode faces, and atomically replace the FAISS index."""
        with self._build_lock:
            database_folder = Path(database_folder).resolve()
            database_folder.mkdir(parents=True, exist_ok=True)
            scanned_image_count = 0
            failed_image_count = 0
            skipped_image_count = 0
            normalized_embeddings: List[np.ndarray] = []
            face_metadata: List[Dict[str, Any]] = []
            exact_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            image_paths = self._list_database_images(database_folder)
            print(f"[face-search] database folder: {database_folder}", flush=True)
            print(f"[face-search] found {len(image_paths)} gallery image(s).", flush=True)
            if not image_paths:
                print("[face-search] no gallery images found. Put images into face_database and restart the service.", flush=True)
            for image_number, image_path in enumerate(image_paths, start=1):
                scanned_image_count += 1
                relative_path = image_path.relative_to(database_folder).as_posix()
                print(f"[face-search] [{image_number}/{len(image_paths)}] encoding: {relative_path}", flush=True)
                try:
                    face_records = self._extract_face_records_from_image(image_path)
                    if not face_records:
                        skipped_image_count += 1
                        print(f"[face-search] [{image_number}/{len(image_paths)}] no face detected: {relative_path}", flush=True)
                        continue
                    print(f"[face-search] [{image_number}/{len(image_paths)}] ready to add {len(face_records)} face vector(s): {relative_path}", flush=True)
                    for face_record in face_records:
                        vector_id = len(face_metadata)
                        normalized_embeddings.append(face_record["normalized_embedding"])
                        face_metadata.append(
                            {
                                "vector_id": vector_id,
                                "file_name": image_path.name,
                                "relative_path": relative_path,
                                "face_index": face_record["face_index"],
                                "face_box": face_record["face_box"],
                            }
                        )
                except Exception as error:
                    failed_image_count += 1
                    print(f"[face-search] failed to process {relative_path}: {error}", flush=True)
            if normalized_embeddings:
                print(f"[face-search] inserting {len(normalized_embeddings)} face vector(s) into FAISS IndexFlatIP...", flush=True)
                embedding_matrix = np.vstack(normalized_embeddings)
                embedding_matrix = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
                exact_index.add(embedding_matrix)
                print(f"[face-search] FAISS insertion finished. ntotal={exact_index.ntotal}", flush=True)
            else:
                print("[face-search] FAISS index is empty because no valid face vector was produced.", flush=True)
            report = IndexBuildReport(
                database_folder=str(database_folder),
                scanned_image_count=scanned_image_count,
                indexed_face_count=len(face_metadata),
                failed_image_count=failed_image_count,
                skipped_image_count=skipped_image_count,
                index_type="faiss.IndexFlatIP exact exhaustive search",
                metric="cosine_similarity over L2-normalized dlib embeddings",
                faiss_thread_count=FAISS_CPU_THREAD_COUNT,
            )
            with self._state_lock:
                self._faiss_index = exact_index
                self._face_metadata = face_metadata
                self._latest_report = report
            print(
                "[face-search] build report: "
                f"scanned={report.scanned_image_count}, indexed_faces={report.indexed_face_count}, "
                f"skipped_no_face={report.skipped_image_count}, failed={report.failed_image_count}",
                flush=True,
            )
            return report

    def get_snapshot(self) -> Tuple[Optional[faiss.IndexFlatIP], List[Dict[str, Any]], IndexBuildReport]:
        """Return a short-lived read snapshot for concurrent search requests."""
        with self._state_lock:
            return self._faiss_index, list(self._face_metadata), self._latest_report

    def get_stats(self) -> Dict[str, Any]:
        """Return current database, model, and FAISS status for UI/API display."""
        faiss_index, face_metadata, report = self.get_snapshot()
        database_folder = Path(report.database_folder)
        gallery_image_count = len(self._list_database_images(database_folder))
        return {
            **asdict(report),
            "database_folder_exists": database_folder.is_dir(),
            "gallery_image_count": gallery_image_count,
            "faiss_ntotal": int(faiss_index.ntotal) if faiss_index is not None else 0,
            "metadata_count": len(face_metadata),
            "detection_model": FACE_DETECTION_MODEL,
            "encoding_model": FACE_ENCODING_MODEL,
            "upsample_times": FACE_DETECTION_UPSAMPLE_TIMES,
            "num_jitters": FACE_ENCODING_NUM_JITTERS,
            "default_top_k": DEFAULT_TOP_K,
            "default_threshold": DEFAULT_MATCH_THRESHOLD,
            "max_upload_mb": MAX_UPLOAD_SIZE_MB,
            "dlib_version": DLIB_VERSION,
            "is_dlib_compiled_with_cuda": IS_DLIB_COMPILED_WITH_CUDA,
            "cuda_device_count": DLIB_CUDA_DEVICE_COUNT,
            "is_cuda_runtime_available": IS_CUDA_RUNTIME_AVAILABLE,
        }

    def search_uploaded_image(self, image_path: Path, top_k: int, threshold: float) -> Dict[str, Any]:
        """Encode one query image and return exact cosine matches sorted high to low."""
        faiss_index, face_metadata, report = self.get_snapshot()
        if faiss_index is None or faiss_index.ntotal <= 0:
            raise RuntimeError(
                "FAISS index is empty. Ask the server administrator to put gallery images into "
                f"{DATABASE_FOLDER} and restart the service."
            )
        query_image = face_recognition.load_image_file(str(image_path))
        query_detect_started_at = time.perf_counter()
        query_face_locations = face_recognition.face_locations(
            query_image,
            number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE_TIMES,
            model=FACE_DETECTION_MODEL,
        )
        query_detect_seconds = time.perf_counter() - query_detect_started_at
        print(f"[face-search] query detection finished: faces={len(query_face_locations)}, seconds={query_detect_seconds:.2f}", flush=True)
        if not query_face_locations:
            raise ValueError("No face detected in query image")
        selected_face_location = max(query_face_locations, key=self._face_area)
        query_encode_started_at = time.perf_counter()
        query_face_encodings = face_recognition.face_encodings(
            query_image,
            known_face_locations=[selected_face_location],
            num_jitters=FACE_ENCODING_NUM_JITTERS,
            model=FACE_ENCODING_MODEL,
        )
        query_encode_seconds = time.perf_counter() - query_encode_started_at
        print(f"[face-search] query encoding finished: encodings={len(query_face_encodings)}, seconds={query_encode_seconds:.2f}", flush=True)
        if not query_face_encodings:
            raise ValueError("Failed to extract query face embedding")
        query_embedding = self._normalize_embedding(query_face_encodings[0])
        indexed_face_count = int(faiss_index.ntotal)
        requested_top_k = int(top_k)
        search_count = indexed_face_count if requested_top_k <= 0 else min(requested_top_k, indexed_face_count)
        faiss_search_started_at = time.perf_counter()
        cosine_similarities, result_indices = faiss_index.search(query_embedding, search_count)
        faiss_search_seconds = time.perf_counter() - faiss_search_started_at
        print(f"[face-search] FAISS search finished: top_k={search_count}, seconds={faiss_search_seconds:.4f}", flush=True)
        results: List[Dict[str, Any]] = []
        for rank_index, vector_index in enumerate(result_indices[0], start=1):
            if vector_index < 0:
                continue
            metadata = dict(face_metadata[int(vector_index)])
            cosine_similarity = float(cosine_similarities[0][rank_index - 1])
            clipped_similarity = max(-1.0, min(1.0, cosine_similarity))
            cosine_distance = float(1.0 - clipped_similarity)
            relative_path = metadata.get("relative_path", "")
            image_url = url_for("serve_database_image", relative_path=relative_path) if relative_path else None
            download_url = url_for("download_database_image", relative_path=relative_path) if relative_path else None
            results.append(
                {
                    "rank": rank_index,
                    "vector_id": metadata.get("vector_id"),
                    "file_name": metadata.get("file_name"),
                    "relative_path": relative_path,
                    "image_url": image_url,
                    "download_url": download_url,
                    "face_index": metadata.get("face_index"),
                    "face_box": metadata.get("face_box"),
                    "cosine_similarity": round(clipped_similarity, 8),
                    "cosine_distance": round(cosine_distance, 8),
                    "similarity_percent": round(clipped_similarity * 100.0, 4),
                    "threshold": threshold,
                    "is_match": bool(clipped_similarity >= threshold),
                }
            )
        return {
            "success": True,
            "query": {
                "file_name": image_path.name,
                "detected_face_count": len(query_face_locations),
                "selected_face_box": self._face_location_to_box(selected_face_location),
            },
            "search": {
                "metric": "cosine_similarity",
                "sort_order": "descending",
                "requested_top_k": requested_top_k,
                "returned_count": len(results),
                "match_count": sum(1 for result in results if result["is_match"]),
                "threshold": threshold,
                "index_type": report.index_type,
                "indexed_face_count": report.indexed_face_count,
                "scanned_image_count": report.scanned_image_count,
            },
            "results": results,
        }


face_search_engine = FaceSearchEngine()

HTML_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>人脸搜索</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root { --red:#e43d18; --orange:#ff7a1a; --dark:#381409; --soft:#fff3ec; --line:#ffd1bd; }
        body { min-height:100vh; background:linear-gradient(145deg,#fff8f4 0%,#ffe7d8 55%,#ffd8c4 100%); color:var(--dark); }
        .panel { background:rgba(255,255,255,.96); border:1px solid rgba(228,61,24,.15); border-radius:22px; box-shadow:0 14px 36px rgba(100,34,10,.10); }
        .btn-theme { background:linear-gradient(135deg,var(--red),var(--orange)); color:#fff; border:0; font-weight:700; }
        .btn-theme:hover { color:#fff; filter:brightness(.98); transform:translateY(-1px); }
        .metric { background:var(--soft); border:1px solid var(--line); border-radius:15px; padding:12px; height:100%; }
        .metric-value { font-size:1.3rem; font-weight:900; color:var(--red); line-height:1; }
        .result-card { border:1px solid rgba(228,61,24,.14); border-radius:18px; background:#fff; overflow:hidden; height:100%; }
        .result-image { display:block; width:100%; height:auto; object-fit:contain; background:#fff7f2; }
        .help-button { position:fixed; right:22px; bottom:22px; width:54px; height:54px; border-radius:50%; border:0; color:#fff; font-size:1.35rem; font-weight:900; background:linear-gradient(135deg,var(--red),var(--orange)); box-shadow:0 14px 28px rgba(228,61,24,.34); z-index:1050; }
        .small-muted { color:#755143; font-size:.92rem; }
        .form-control:focus { border-color:var(--orange); box-shadow:0 0 0 .25rem rgba(255,122,26,.18); }
        code { color:#a92a10; }
    </style>
</head>
<body>
<main class="container py-4 py-lg-5">
    <section class="panel p-4 mb-4">
        <form id="searchForm" class="row g-3 align-items-end">
            <div class="col-lg-6">
                <label class="form-label fw-bold">上传查询图片</label>
                <input class="form-control form-control-lg" type="file" name="image" accept="image/*" required>
            </div>
            <div class="col-sm-6 col-lg-2">
                <label class="form-label fw-bold">返回数量</label>
                <input class="form-control" type="number" name="top_k" min="0" value="0">
            </div>
            <div class="col-sm-6 col-lg-2">
                <label class="form-label fw-bold">匹配阈值</label>
                <input class="form-control" type="number" name="threshold" min="-1" max="1" step="0.01" value="0.60">
            </div>
            <div class="col-lg-2 d-grid">
                <button class="btn btn-theme btn-lg" type="submit">搜索</button>
            </div>
            <div class="col-12 small-muted">返回数量填 0 或留空表示返回全部结果；图片只用于本次查询，不会写入数据库。</div>
        </form>
    </section>
    <section id="messageArea"></section>
    <section id="statsArea" class="row g-3 mb-4"></section>
    <section id="summaryArea" class="mb-4"></section>
    <section id="resultsArea" class="row g-4"></section>
</main>

<button class="help-button" type="button" data-bs-toggle="modal" data-bs-target="#apiHelpModal">?</button>
<div class="modal fade" id="apiHelpModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-centered"><div class="modal-content rounded-4">
        <div class="modal-header border-0"><h5 class="modal-title fw-bold">API</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
        <div class="modal-body">
            <p><code>POST /api/search</code></p>
            <pre class="bg-light rounded-4 p-3"><code>curl -k -X POST https://localhost:5000/api/search \
  -F "image=@/path/to/query.jpg" \
  -F "top_k=0" \
  -F "threshold=0.60"</code></pre>
            <p><code>GET /api/stats</code></p>
            <p class="mb-0"><code>GET /api/health</code></p>
        </div>
    </div></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
const statsArea = document.getElementById('statsArea');
const messageArea = document.getElementById('messageArea');
const summaryArea = document.getElementById('summaryArea');
const resultsArea = document.getElementById('resultsArea');
const searchForm = document.getElementById('searchForm');

function escapeHtml(value) {
    // Escape text before rendering file/API values as HTML.
    return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function showMessage(type, text) {
    // Show the latest user-facing operation message.
    messageArea.innerHTML = `<div class="alert alert-${type} rounded-4 shadow-sm">${text}</div>`;
}

async function loadStats() {
    // Load current FAISS and gallery status from the backend.
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        const ready = data.indexed_face_count > 0;
        statsArea.innerHTML = `
            <div class="col-md-3"><div class="metric"><div class="small-muted">状态</div><div class="metric-value">${ready ? 'READY' : 'EMPTY'}</div></div></div>
            <div class="col-md-3"><div class="metric"><div class="small-muted">图库图片</div><div class="metric-value">${data.gallery_image_count}</div></div></div>
            <div class="col-md-3"><div class="metric"><div class="small-muted">入库人脸</div><div class="metric-value">${data.indexed_face_count}</div></div></div>
            <div class="col-md-3"><div class="metric"><div class="small-muted">FAISS 向量</div><div class="metric-value">${data.faiss_ntotal}</div></div></div>`;
    } catch (error) {
        statsArea.innerHTML = '<div class="col-12 text-danger">无法读取服务状态。</div>';
    }
}

function renderSummary(data) {
    // Render search summary cards.
    summaryArea.innerHTML = `<div class="panel p-4"><div class="row g-3">
        <div class="col-md-3"><div class="metric"><div class="small-muted">返回结果</div><div class="metric-value">${data.search.returned_count}</div></div></div>
        <div class="col-md-3"><div class="metric"><div class="small-muted">阈值命中</div><div class="metric-value">${data.search.match_count}</div></div></div>
        <div class="col-md-3"><div class="metric"><div class="small-muted">阈值</div><div class="metric-value">${data.search.threshold}</div></div></div>
        <div class="col-md-3"><div class="metric"><div class="small-muted">查询图人脸</div><div class="metric-value">${data.query.detected_face_count}</div></div></div>
    </div></div>`;
}

function renderResults(results) {
    // Render ranked face match results, with full image display and download links.
    if (!results.length) {
        resultsArea.innerHTML = '<div class="col-12"><div class="alert alert-warning rounded-4">没有返回结果。</div></div>';
        return;
    }
    resultsArea.innerHTML = results.map(result => `<div class="col-md-6 col-xl-4"><div class="result-card">
        ${result.image_url ? `<a href="${result.image_url}" target="_blank"><img class="result-image" src="${result.image_url}" alt="matched image"></a>` : ''}
        <div class="p-3">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span class="badge rounded-pill text-bg-danger">Top ${result.rank}</span>
                <span class="badge ${result.is_match ? 'text-bg-success' : 'text-bg-secondary'} rounded-pill">${result.is_match ? 'MATCH' : 'NO MATCH'}</span>
            </div>
            <h5 class="fw-bold text-break mb-2">${escapeHtml(result.file_name)}</h5>
            <div class="small-muted text-break mb-3">${escapeHtml(result.relative_path)}</div>
            <div class="row g-2 small mb-3">
                <div class="col-6"><strong>相似度</strong><br>${result.similarity_percent}%</div>
                <div class="col-6"><strong>Cosine</strong><br>${result.cosine_similarity}</div>
                <div class="col-6"><strong>Distance</strong><br>${result.cosine_distance}</div>
                <div class="col-6"><strong>Face Index</strong><br>${result.face_index}</div>
            </div>
            <div class="d-flex flex-wrap gap-2">
                ${result.image_url ? `<a class="btn btn-sm btn-outline-danger" href="${result.image_url}" target="_blank">查看完整图片</a>` : ''}
                ${result.download_url ? `<a class="btn btn-sm btn-theme" href="${result.download_url}">下载图片</a>` : ''}
            </div>
        </div>
    </div></div>`).join('');
}

async function submitSearch(event) {
    // Upload one query image and render exact cosine-similarity search results.
    event.preventDefault();
    summaryArea.innerHTML = '';
    resultsArea.innerHTML = '';
    showMessage('warning', '正在搜索...');
    const formData = new FormData(searchForm);
    if (!formData.get('top_k')) formData.set('top_k', '0');
    try {
        const response = await fetch('/api/search', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || '搜索失败');
        showMessage('success', '搜索完成。');
        renderSummary(data);
        renderResults(data.results);
        loadStats();
    } catch (error) {
        showMessage('danger', escapeHtml(error.message));
        loadStats();
    }
}

searchForm.addEventListener('submit', submitSearch);
loadStats();
</script>
</body>
</html>
"""

def parse_top_k(raw_value: Optional[str]) -> int:
    """Parse the top_k form/API value; 0 means return all indexed faces."""
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_TOP_K
    try:
        parsed_value = int(str(raw_value).strip())
    except ValueError as error:
        raise ValueError("top_k must be an integer. Use 0 to return all results.") from error
    if parsed_value < 0:
        raise ValueError("top_k must be greater than or equal to 0")
    return parsed_value


def parse_threshold(raw_value: Optional[str]) -> float:
    """Parse and validate the cosine-similarity match threshold."""
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_MATCH_THRESHOLD
    try:
        parsed_value = float(str(raw_value).strip())
    except ValueError as error:
        raise ValueError("threshold must be a number") from error
    if parsed_value < -1.0 or parsed_value > 1.0:
        raise ValueError("threshold must be between -1.0 and 1.0 for cosine similarity")
    return parsed_value


def validate_image_filename(file_name: str) -> str:
    """Return a safe query image filename or raise if the extension is unsupported."""
    safe_name = secure_filename(file_name) or "image.jpg"
    file_extension = Path(safe_name).suffix.lower()
    if file_extension not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {file_extension}")
    return safe_name


def save_query_image() -> Path:
    """Save the uploaded query image into the temporary upload folder."""
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        raise ValueError("Missing query image. Upload using form field name: image")
    safe_name = validate_image_filename(uploaded_file.filename)
    saved_path = UPLOAD_FOLDER / f"query_{uuid.uuid4().hex}{Path(safe_name).suffix.lower()}"
    uploaded_file.save(saved_path)
    return saved_path


@app.route("/", methods=["GET"])
def index_page():
    """Serve the embedded Bootstrap search-only web interface."""
    return render_template_string(HTML_PAGE, database_folder=str(DATABASE_FOLDER))


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return current database and FAISS status."""
    return jsonify(face_search_engine.get_stats())


@app.route("/api/health", methods=["GET"])
def api_health():
    """Return a lightweight readiness response for cloud health checks."""
    stats = face_search_engine.get_stats()
    return jsonify(
        {
            "success": True,
            "ready": stats["indexed_face_count"] > 0,
            "indexed_face_count": stats["indexed_face_count"],
            "gallery_image_count": stats["gallery_image_count"],
            "metric": stats["metric"],
            "index_type": stats["index_type"],
        }
    )


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search one uploaded query image against the startup-built FAISS face database."""
    saved_path: Optional[Path] = None
    try:
        top_k = parse_top_k(request.form.get("top_k"))
        threshold = parse_threshold(request.form.get("threshold"))
        saved_path = save_query_image()
        return jsonify(face_search_engine.search_uploaded_image(saved_path, top_k, threshold))
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 400
    finally:
        if saved_path is not None:
            try:
                saved_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.route("/database-image/<path:relative_path>", methods=["GET"])
def serve_database_image(relative_path: str):
    """Serve matched gallery images from face_database for full result display."""
    return send_from_directory(DATABASE_FOLDER, relative_path)


@app.route("/database-image-download/<path:relative_path>", methods=["GET"])
def download_database_image(relative_path: str):
    """Download the matched gallery image as an attachment."""
    return send_from_directory(DATABASE_FOLDER, relative_path, as_attachment=True)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):
    """Return a clear JSON error when the query image exceeds the size limit."""
    return jsonify({"success": False, "error": f"Uploaded file is larger than {MAX_UPLOAD_SIZE_MB} MB"}), 413

if __name__ == "__main__":
    print_runtime_configuration()
    print("[face-search] starting service and building initial FAISS index...", flush=True)
    startup_report = face_search_engine.rebuild_index(DATABASE_FOLDER)
    print(
        "[face-search] ready: "
        f"{startup_report.indexed_face_count} face vector(s) from "
        f"{startup_report.scanned_image_count} image(s). "
        f"Skipped no-face: {startup_report.skipped_image_count}. "
        f"Failed: {startup_report.failed_image_count}. "
        "Open https://localhost:5000",
        flush=True,
    )
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context="adhoc")
