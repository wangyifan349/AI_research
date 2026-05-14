"""
face_search.py

This file builds a face index from all images in a directory.
It supports searching one query face image against the indexed faces.
Results are sorted by cosine similarity in descending order.
"""

import os  # Used for directory traversal and file path handling
import pickle  # Used to save and load face index cache
import numpy as np  # Used for vector calculation
import face_recognition  # Face recognition library based on dlib


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # Supported image extensions


def is_image_file(file_path):  # Check whether a file is a supported image
    file_extension = os.path.splitext(file_path)[1].lower()  # Get lowercase file extension
    return file_extension in IMAGE_EXTENSIONS  # Return True if file extension is supported


def cosine_similarity(vector_a, vector_b):  # Calculate cosine similarity between two vectors
    vector_a = np.asarray(vector_a, dtype=np.float32)  # Convert first vector to numpy array
    vector_b = np.asarray(vector_b, dtype=np.float32)  # Convert second vector to numpy array

    norm_a = np.linalg.norm(vector_a)  # Calculate first vector norm
    norm_b = np.linalg.norm(vector_b)  # Calculate second vector norm

    if norm_a == 0 or norm_b == 0:  # Avoid division by zero
        return 0.0  # Return 0 for invalid vectors

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))  # Return cosine similarity


def get_face_area(face_location):  # Calculate face box area
    top, right, bottom, left = face_location  # Unpack dlib face box
    return (right - left) * (bottom - top)  # Return width * height


def extract_face_encodings_from_image(
    image_path,
    detector_model="cnn",
    encoding_model="large",
    upsample_times=1,
    jitter_times=1
):
    image = face_recognition.load_image_file(image_path)  # Load image from file path

    face_locations = face_recognition.face_locations(
        image,  # Input image
        number_of_times_to_upsample=upsample_times,  # Increase this value to detect smaller faces
        model=detector_model  # Use "cnn" for accuracy or "hog" for speed
    )

    if not face_locations:  # Check whether any face was detected
        return []  # Return empty list if no face exists

    face_encodings = face_recognition.face_encodings(
        image,  # Input image
        known_face_locations=face_locations,  # Use all detected face locations
        num_jitters=jitter_times,  # More jitters may improve stability but reduce speed
        model=encoding_model  # Use "large" for accuracy or "small" for speed
    )

    face_records = []  # Store all faces extracted from this image

    for face_index, face_encoding in enumerate(face_encodings):  # Iterate each extracted face
        face_location = face_locations[face_index]  # Get matching face location

        face_records.append({
            "image_path": image_path,  # Source image path
            "face_index": face_index,  # Face index in the source image
            "face_encoding": face_encoding,  # 128-dimensional face embedding
            "face_area": get_face_area(face_location),  # Face box area
            "face_location": {  # Face box coordinates
                "top": face_location[0],  # Top coordinate
                "right": face_location[1],  # Right coordinate
                "bottom": face_location[2],  # Bottom coordinate
                "left": face_location[3]  # Left coordinate
            }
        })

    return face_records  # Return all face records from this image


def build_face_index(
    image_root_directory,
    detector_model="cnn",
    encoding_model="large",
    upsample_times=1,
    jitter_times=1
):
    face_index = []  # Store all indexed face records

    for current_directory, _, file_names in os.walk(image_root_directory):  # Walk through directory recursively
        for file_name in file_names:  # Iterate every file in current directory
            image_path = os.path.join(current_directory, file_name)  # Build full image path

            if not is_image_file(image_path):  # Skip unsupported files
                continue  # Continue next file

            image_face_records = extract_face_encodings_from_image(
                image_path=image_path,  # Current image path
                detector_model=detector_model,  # Face detection model
                encoding_model=encoding_model,  # Face encoding model
                upsample_times=upsample_times,  # Face detection upsample count
                jitter_times=jitter_times  # Face encoding jitter count
            )

            face_index.extend(image_face_records)  # Add extracted faces into index

    return face_index  # Return complete face index


def save_face_index(face_index, index_cache_path):  # Save face index to local cache
    with open(index_cache_path, "wb") as cache_file:  # Open cache file in binary write mode
        pickle.dump(face_index, cache_file)  # Save face index using pickle


def load_face_index(index_cache_path):  # Load face index from local cache
    with open(index_cache_path, "rb") as cache_file:  # Open cache file in binary read mode
        return pickle.load(cache_file)  # Return loaded face index


def search_face_by_image(
    query_image_path,
    face_index,
    top_k=10,
    cosine_threshold=0.93,
    detector_model="cnn",
    encoding_model="large",
    upsample_times=1,
    jitter_times=1
):
    query_face_records = extract_face_encodings_from_image(
        image_path=query_image_path,  # Query face image path
        detector_model=detector_model,  # Face detection model
        encoding_model=encoding_model,  # Face encoding model
        upsample_times=upsample_times,  # Face detection upsample count
        jitter_times=jitter_times  # Face encoding jitter count
    )

    if not query_face_records:  # Check whether query image contains a valid face
        raise ValueError(f"No face detected in query image: {query_image_path}")  # Stop if query face is missing

    query_face_record = max(query_face_records, key=lambda item: item["face_area"])  # Use largest face as query face
    query_face_encoding = query_face_record["face_encoding"]  # Get query face embedding

    search_results = []  # Store all comparison results

    for indexed_face_record in face_index:  # Compare query face with each indexed face
        indexed_face_encoding = indexed_face_record["face_encoding"]  # Get indexed face embedding

        cosine_score = cosine_similarity(query_face_encoding, indexed_face_encoding)  # Calculate cosine similarity
        euclidean_distance = float(np.linalg.norm(query_face_encoding - indexed_face_encoding))  # Calculate distance

        search_results.append({
            "image_path": indexed_face_record["image_path"],  # Matched image path
            "face_index": indexed_face_record["face_index"],  # Matched face index in image
            "is_same_person": cosine_score >= cosine_threshold,  # Match decision based on cosine threshold
            "cosine_similarity": round(cosine_score, 6),  # Cosine similarity, higher means more similar
            "cosine_percent": round(cosine_score * 100, 2),  # Cosine similarity percentage
            "euclidean_distance": round(euclidean_distance, 6),  # Euclidean distance, lower means more similar
            "face_location": indexed_face_record["face_location"]  # Matched face location
        })

    sorted_results = sorted(
        search_results,  # All search results
        key=lambda item: (-item["cosine_similarity"], item["euclidean_distance"])  # Sort by cosine desc, distance asc
    )

    return sorted_results[:top_k]  # Return top K most similar results


def search_face_in_directory(
    query_image_path,
    image_root_directory,
    top_k=10,
    index_cache_path=None,
    rebuild_index=False,
    cosine_threshold=0.93,
    detector_model="cnn",
    encoding_model="large",
    upsample_times=1,
    jitter_times=1
):
    if index_cache_path and os.path.exists(index_cache_path) and not rebuild_index:  # Use cache if allowed
        face_index = load_face_index(index_cache_path)  # Load existing face index
    else:
        face_index = build_face_index(
            image_root_directory=image_root_directory,  # Directory containing images to search
            detector_model=detector_model,  # Face detection model
            encoding_model=encoding_model,  # Face encoding model
            upsample_times=upsample_times,  # Face detection upsample count
            jitter_times=jitter_times  # Face encoding jitter count
        )

        if index_cache_path:  # Check whether cache path is provided
            save_face_index(face_index, index_cache_path)  # Save new face index to cache

    return search_face_by_image(
        query_image_path=query_image_path,  # Image used as search query
        face_index=face_index,  # Face index built from directory
        top_k=top_k,  # Return only the top K most similar faces
        cosine_threshold=cosine_threshold,  # Similarity threshold for same-person decision
        detector_model=detector_model,  # Face detection model
        encoding_model=encoding_model,  # Face encoding model
        upsample_times=upsample_times,  # Face detection upsample count
        jitter_times=jitter_times  # Face encoding jitter count
    )


if __name__ == "__main__":  # Run this block only when executing this file directly
    results = search_face_in_directory(
        query_image_path="query.jpg",  # The face image you want to search
        image_root_directory="./faces",  # Directory containing searchable face images
        top_k=5,  # Return the top 5 most similar results
        index_cache_path="face_index.pkl",  # Cache file for saved face embeddings
        rebuild_index=True,  # True means rebuild index, False means reuse cache if it exists
        cosine_threshold=0.93,  # Minimum cosine similarity to mark as same person
        detector_model="cnn",  # "cnn" is more accurate, "hog" is faster
        encoding_model="large",  # "large" is more accurate, "small" is faster
        upsample_times=1,  # Increase this value to detect smaller faces
        jitter_times=1  # Increase this value for more stable embeddings but slower speed
    )

    for result in results:  # Print each search result
        print(result)  # Print one matched face record
