"""
1:N Face Search System with FAISS Exact Search + float64 Final Rerank
(dlib / face_recognition version)
Accuracy policy:
- FAISS uses faiss.IndexFlatL2: exact exhaustive L2 search.
- No IVF, PQ, HNSW, quantization, compression, clustering, or approximate search.
- No normalization, transformation, or compression of face vectors.
- Original database embeddings are preserved as float64.
- Original query embeddings are preserved as float64.
- FAISS receives float32 copies only because FAISS standard input is float32.
- Final Top-K ranking, displayed distance, and match decision are computed with float64 NumPy Euclidean distance.
"""
import os
import time
from typing import Dict, List, Optional, Tuple

import dlib
import faiss
import face_recognition
import numpy as np
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # Supported database image extensions.
TOP_K = 10                                                     # Number of final results to print.
MATCH_THRESHOLD = 0.45                                         # Euclidean distance threshold; smaller is stricter.
DETECTION_MODEL_POLICY = "auto"                                # "auto", "hog", or "cnn"; auto uses cnn only when CUDA is available.
ENCODING_MODEL = "large"                                       # dlib face encoding model: "small" or "large".
UPSAMPLE_TIMES = None                                          # None means auto: 0 for CNN, 1 for HOG; use an integer to force.
JITTER_COUNT = 2                                               # Encoding jitter count; increase for stability, decrease for speed.
EMBEDDING_DIMENSION = 128                                      # face_recognition / dlib embedding dimension.
FAISS_THREAD_COUNT = max(1, os.cpu_count() or 1)                # FAISS CPU thread count for exact search.
FINAL_RERANK_SCOPE = "all"                                     # "all" = strict full float64 rerank; "faiss_candidates" = faster candidate rerank.
FAISS_CANDIDATE_MULTIPLIER = 10                                # Used only when FINAL_RERANK_SCOPE == "faiss_candidates".
MIN_FAISS_CANDIDATES = 100                                     # Minimum FAISS candidates for candidate rerank mode.

Metadata = Dict[str, object]

def get_dlib_cuda_device_count() -> int:
    """Return the number of CUDA devices visible to dlib, safely handling CPU-only builds."""
    try:
        return int(dlib.cuda.get_num_devices())
    except Exception:
        return 0

def resolve_detection_model() -> str:
    """Choose the dlib face detector from configuration and current CUDA availability."""
    requested_model = DETECTION_MODEL_POLICY.lower().strip()
    cuda_is_usable = bool(dlib.DLIB_USE_CUDA and get_dlib_cuda_device_count() > 0)
    if requested_model == "auto":
        return "cnn" if cuda_is_usable else "hog"
    if requested_model in {"hog", "cnn"}:
        return requested_model
    raise ValueError('DETECTION_MODEL_POLICY must be "auto", "hog", or "cnn".')

def resolve_upsample_times(selected_detection_model: str) -> int:
    """Choose the face-detection upsample count from configuration and selected detector."""
    if UPSAMPLE_TIMES is not None:
        return int(UPSAMPLE_TIMES)
    return 0 if selected_detection_model == "cnn" else 1

def print_runtime_status(selected_detection_model: str, selected_upsample_times: int) -> None:
    """Print dlib, CUDA, FAISS, and model-selection status before the search loop starts."""
    cuda_device_count = get_dlib_cuda_device_count()
    cuda_is_usable = bool(dlib.DLIB_USE_CUDA and cuda_device_count > 0)
    print("\n=== RUNTIME STATUS ===", flush=True)
    print(f"dlib version: {dlib.__version__}", flush=True)
    print(f"dlib compiled with CUDA: {bool(dlib.DLIB_USE_CUDA)}", flush=True)
    print(f"dlib CUDA device count: {cuda_device_count}", flush=True)
    print(f"CUDA runtime available: {cuda_is_usable}", flush=True)
    print(f"Configured detection policy: {DETECTION_MODEL_POLICY}", flush=True)
    print(f"Selected detection model: {selected_detection_model}", flush=True)
    print(f"Selected encoding model: {ENCODING_MODEL}", flush=True)
    print(f"Detection upsample times: {selected_upsample_times}", flush=True)
    print(f"Encoding jitter count: {JITTER_COUNT}", flush=True)
    print(f"FAISS CPU threads: {FAISS_THREAD_COUNT}", flush=True)
    print(f"Final rerank scope: {FINAL_RERANK_SCOPE}", flush=True)
    print("FAISS index type: IndexFlatL2 exact exhaustive search", flush=True)
    print("Final distance type: NumPy float64 Euclidean distance", flush=True)
    if selected_detection_model == "cnn" and not cuda_is_usable:
        print("[WARN] CNN detector is selected without usable CUDA. It may be very slow on CPU.", flush=True)
    print("=" * 80, flush=True)

def is_supported_image(file_name: str) -> bool:
    """Return True when the file has a supported image extension."""
    return os.path.splitext(file_name.lower())[1] in IMAGE_EXTENSIONS

def detect_faces(image_array: np.ndarray, detection_model: str, upsample_times: int) -> List[Tuple[int, int, int, int]]:
    """Detect face boxes in one image and return face_recognition location tuples."""
    return face_recognition.face_locations(
        image_array,
        model=detection_model,
        number_of_times_to_upsample=upsample_times,
    )

def encode_faces(image_array: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """Extract dlib embeddings for detected face locations."""
    return face_recognition.face_encodings(
        image_array,
        known_face_locations=face_locations,
        num_jitters=JITTER_COUNT,
        model=ENCODING_MODEL,
    )

def load_query_embedding_float64(query_image_path: str, detection_model: str, upsample_times: int) -> Optional[np.ndarray]:
    """Load the query image, detect the first face, and return one original float64 embedding with shape (1, 128)."""
    query_image = face_recognition.load_image_file(query_image_path)
    detection_started_at = time.perf_counter()
    query_face_locations = detect_faces(query_image, detection_model, upsample_times)
    detection_seconds = time.perf_counter() - detection_started_at
    print(f"[query] detected {len(query_face_locations)} face(s) in {detection_seconds:.2f}s", flush=True)
    if not query_face_locations:
        print("No face detected in query image", flush=True)
        return None
    encoding_started_at = time.perf_counter()
    query_face_encodings = encode_faces(query_image, [query_face_locations[0]])
    encoding_seconds = time.perf_counter() - encoding_started_at
    print(f"[query] encoded {len(query_face_encodings)} face(s) in {encoding_seconds:.2f}s", flush=True)
    if not query_face_encodings:
        print("Failed to extract query face embedding", flush=True)
        return None
    return np.asarray(query_face_encodings[0], dtype=np.float64).reshape(1, EMBEDDING_DIMENSION)

def build_exact_faiss_database(face_database_folder: str, detection_model: str, upsample_times: int):
    """Scan database images, extract embeddings, preserve float64 vectors, and build exact FAISS IndexFlatL2 from float32 copies."""
    database_embeddings: List[np.ndarray] = []
    database_metadata: List[Metadata] = []
    scanned_image_count = 0
    indexed_face_count = 0
    failed_image_count = 0
    skipped_image_count = 0
    scan_started_at = time.perf_counter()
    for current_folder, _, file_names in os.walk(face_database_folder):
        for file_name in sorted(file_names):
            if not is_supported_image(file_name):
                continue
            image_path = os.path.join(current_folder, file_name)
            scanned_image_count += 1
            image_started_at = time.perf_counter()
            print(f"[database] scanning #{scanned_image_count}: {image_path}", flush=True)
            try:
                database_image = face_recognition.load_image_file(image_path)
                detection_started_at = time.perf_counter()
                database_face_locations = detect_faces(database_image, detection_model, upsample_times)
                detection_seconds = time.perf_counter() - detection_started_at
                print(f"[database] detected {len(database_face_locations)} face(s) in {detection_seconds:.2f}s", flush=True)
                if not database_face_locations:
                    skipped_image_count += 1
                    continue
                encoding_started_at = time.perf_counter()
                database_face_encodings = encode_faces(database_image, database_face_locations)
                encoding_seconds = time.perf_counter() - encoding_started_at
                print(f"[database] encoded {len(database_face_encodings)} face(s) in {encoding_seconds:.2f}s", flush=True)
                for face_index, face_embedding in enumerate(database_face_encodings):
                    face_embedding_float64 = np.asarray(face_embedding, dtype=np.float64)
                    if face_embedding_float64.shape != (EMBEDDING_DIMENSION,):
                        print(f"[WARN] Invalid embedding shape {face_embedding_float64.shape}; skipped: {image_path}", flush=True)
                        continue
                    database_embeddings.append(face_embedding_float64)
                    database_metadata.append({
                        "file_name": file_name,
                        "folder_path": current_folder,
                        "file_path": image_path,
                        "face_index": face_index,
                    })
                    indexed_face_count += 1
            except Exception as error:
                failed_image_count += 1
                print(f"[ERROR] Failed to process image: {image_path} | {error}", flush=True)
            image_seconds = time.perf_counter() - image_started_at
            print(f"[database] finished in {image_seconds:.2f}s: {image_path}", flush=True)
    if not database_embeddings:
        return None, None, database_metadata, scanned_image_count, indexed_face_count, skipped_image_count, failed_image_count
    matrix_started_at = time.perf_counter()
    database_embeddings_float64 = np.vstack(database_embeddings).astype(np.float64, copy=False)
    database_embeddings_float32 = database_embeddings_float64.astype(np.float32)
    matrix_seconds = time.perf_counter() - matrix_started_at
    insertion_started_at = time.perf_counter()
    exact_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)        # Exact exhaustive squared-L2 index; not approximate.
    exact_index.add(database_embeddings_float32)                # FAISS receives float32 copies; original float64 vectors are preserved.
    insertion_seconds = time.perf_counter() - insertion_started_at
    scan_seconds = time.perf_counter() - scan_started_at
    print(f"[database] built float64 matrix shape: {database_embeddings_float64.shape}", flush=True)
    print(f"[database] prepared float32 FAISS copy in {matrix_seconds:.4f}s", flush=True)
    print(f"[faiss] inserted {exact_index.ntotal} vector(s) in {insertion_seconds:.4f}s", flush=True)
    print(f"[database] total build time: {scan_seconds:.2f}s", flush=True)
    return exact_index, database_embeddings_float64, database_metadata, scanned_image_count, indexed_face_count, skipped_image_count, failed_image_count

def compute_float64_euclidean_distances(database_embeddings_float64: np.ndarray, query_embedding_float64: np.ndarray) -> np.ndarray:
    """Compute ordinary Euclidean L2 distances using float64 NumPy."""
    return np.linalg.norm(database_embeddings_float64 - query_embedding_float64, axis=1)

def search_with_faiss_then_float64_rerank(exact_index, database_embeddings_float64: np.ndarray, query_embedding_float64: np.ndarray, top_k: int):
    """Search with FAISS, then compute final result order and distances with float64 NumPy."""
    total_vectors = int(exact_index.ntotal)
    effective_top_k = min(top_k, total_vectors)
    query_embedding_float32 = query_embedding_float64.astype(np.float32)
    if FINAL_RERANK_SCOPE == "all":
        faiss_search_count = total_vectors
    elif FINAL_RERANK_SCOPE == "faiss_candidates":
        faiss_search_count = min(total_vectors, max(top_k * FAISS_CANDIDATE_MULTIPLIER, MIN_FAISS_CANDIDATES))
    else:
        raise ValueError('FINAL_RERANK_SCOPE must be "all" or "faiss_candidates".')
    faiss_started_at = time.perf_counter()
    faiss_squared_distances, faiss_indices = exact_index.search(query_embedding_float32, faiss_search_count)
    faiss_seconds = time.perf_counter() - faiss_started_at
    print(f"[faiss] exact IndexFlatL2 search returned {faiss_search_count} candidate(s) in {faiss_seconds:.6f}s", flush=True)
    rerank_started_at = time.perf_counter()
    if FINAL_RERANK_SCOPE == "all":
        all_float64_distances = compute_float64_euclidean_distances(database_embeddings_float64, query_embedding_float64)
        sorted_indices = np.argsort(all_float64_distances, kind="mergesort")
        final_indices = sorted_indices[:effective_top_k]
        final_distances = all_float64_distances[final_indices]
    else:
        candidate_indices = faiss_indices[0]
        candidate_indices = candidate_indices[candidate_indices >= 0].astype(np.int64)
        candidate_embeddings_float64 = database_embeddings_float64[candidate_indices]
        candidate_float64_distances = compute_float64_euclidean_distances(candidate_embeddings_float64, query_embedding_float64)
        candidate_order = np.argsort(candidate_float64_distances, kind="mergesort")
        final_candidate_order = candidate_order[:effective_top_k]
        final_indices = candidate_indices[final_candidate_order]
        final_distances = candidate_float64_distances[final_candidate_order]
    rerank_seconds = time.perf_counter() - rerank_started_at
    print(f"[rerank] final float64 rerank finished in {rerank_seconds:.6f}s", flush=True)
    return final_indices, final_distances, faiss_squared_distances, faiss_indices

def print_search_results(
    query_image_path: str,
    face_database_folder: str,
    database_metadata: List[Metadata],
    scanned_image_count: int,
    indexed_face_count: int,
    skipped_image_count: int,
    failed_image_count: int,
    final_indices: np.ndarray,
    final_distances: np.ndarray,
) -> None:
    """Print final ranked results using float64 Euclidean distances."""
    print("\n=== FAISS EXACT SEARCH + FLOAT64 FINAL RERANK RESULTS ===")
    print(f"Query image: {query_image_path}")
    print(f"Face database folder: {face_database_folder}")
    print(f"Scanned images: {scanned_image_count}")
    print(f"Indexed faces: {indexed_face_count}")
    print(f"Skipped no-face images: {skipped_image_count}")
    print(f"Failed images: {failed_image_count}")
    print("FAISS index type: IndexFlatL2 exact exhaustive search")
    print(f"Final rerank scope: {FINAL_RERANK_SCOPE}")
    print("Final distance: float64 Euclidean distance")
    print("Distance rule: smaller is more similar")
    print(f"Match threshold: {MATCH_THRESHOLD:.8f}")
    print("-" * 80)
    for rank, vector_index in enumerate(final_indices, start=1):
        metadata = database_metadata[int(vector_index)]
        euclidean_distance = float(final_distances[rank - 1])
        squared_l2_distance = euclidean_distance * euclidean_distance
        display_similarity = max(0.0, 1.0 - euclidean_distance)  # Display-only score; not used for matching.
        is_match = euclidean_distance <= MATCH_THRESHOLD          # Final match decision uses float64 Euclidean distance.
        print(f"Top {rank}")
        print(f"  File name: {metadata['file_name']}")
        print(f"  Folder path: {metadata['folder_path']}")
        print(f"  Full file path: {metadata['file_path']}")
        print(f"  Face index in image: {metadata['face_index']}")
        print(f"  Final squared L2 distance: {squared_l2_distance:.12f}")
        print(f"  Final Euclidean distance: {euclidean_distance:.12f}")
        print(f"  Display similarity: {display_similarity:.12f}")
        print(f"  Match: {is_match}")
        print("-" * 80)

def main() -> None:
    """Run the interactive command-line face search loop."""
    faiss.omp_set_num_threads(FAISS_THREAD_COUNT)                # Apply FAISS CPU thread count.
    selected_detection_model = resolve_detection_model()
    selected_upsample_times = resolve_upsample_times(selected_detection_model)
    print_runtime_status(selected_detection_model, selected_upsample_times)
    face_database_folder = input("Enter face database folder path: ").strip().strip('"').strip("'")
    if not os.path.isdir(face_database_folder):
        raise SystemExit("Invalid folder path")
    print("\nBuilding face database once...", flush=True)
    (
        exact_index,
        database_embeddings_float64,
        database_metadata,
        scanned_image_count,
        indexed_face_count,
        skipped_image_count,
        failed_image_count,
    ) = build_exact_faiss_database(face_database_folder, selected_detection_model, selected_upsample_times)
    if exact_index is None or database_embeddings_float64 is None or exact_index.ntotal <= 0:
        raise SystemExit("No valid database face embeddings found")
    print("\nDatabase is ready. You can now enter query image paths.", flush=True)
    while True:
        query_image_path = input("\nEnter query image path (q to quit): ").strip().strip('"').strip("'")
        if query_image_path.lower() in {"q", "quit", "exit"}:
            break
        if not os.path.isfile(query_image_path):
            print("Invalid query image path", flush=True)
            continue
        query_embedding_float64 = load_query_embedding_float64(query_image_path, selected_detection_model, selected_upsample_times)
        if query_embedding_float64 is None:
            continue
        final_indices, final_distances, _, _ = search_with_faiss_then_float64_rerank(
            exact_index=exact_index,
            database_embeddings_float64=database_embeddings_float64,
            query_embedding_float64=query_embedding_float64,
            top_k=TOP_K,
        )
        print_search_results(
            query_image_path=query_image_path,
            face_database_folder=face_database_folder,
            database_metadata=database_metadata,
            scanned_image_count=scanned_image_count,
            indexed_face_count=indexed_face_count,
            skipped_image_count=skipped_image_count,
            failed_image_count=failed_image_count,
            final_indices=final_indices,
            final_distances=final_distances,
        )

if __name__ == "__main__":
    main()
