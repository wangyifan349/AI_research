"""
face_compare.py

This file compares two face images using face_recognition/dlib.
It extracts 128-dimensional face embeddings, calculates cosine similarity,
calculates Euclidean distance, and returns a simple comparison result.
"""
import os  # Used to check whether image files exist
import numpy as np  # Used for vector and distance calculation
import face_recognition  # Face recognition library based on dlib
def cosine_similarity(vec1, vec2):  # Calculate cosine similarity between two vectors
    vec1 = np.asarray(vec1, dtype=np.float32)  # Convert first vector to numpy array
    vec2 = np.asarray(vec2, dtype=np.float32)  # Convert second vector to numpy array
    norm1 = np.linalg.norm(vec1)  # Calculate first vector norm
    norm2 = np.linalg.norm(vec2)  # Calculate second vector norm
    if norm1 == 0 or norm2 == 0:  # Avoid division by zero
        return 0.0  # Return 0 for invalid vectors
    return float(np.dot(vec1, vec2) / (norm1 * norm2))  # Return cosine similarity


def get_largest_face(face_locations):  # Select the largest face from detected faces
    return max(
        face_locations,  # All detected face boxes
        key=lambda box: (box[1] - box[3]) * (box[2] - box[0])  # Face area: width * height
    )

def get_face_encoding(
    image_path,
    detector_model="cnn",
    encoding_model="large",
    upsample=1,
    num_jitters=1
):
    if not os.path.exists(image_path):  # Check whether image file exists
        raise FileNotFoundError(f"Image file not found: {image_path}")  # Raise error if file does not exist
    image = face_recognition.load_image_file(image_path)  # Load image from file path
    face_locations = face_recognition.face_locations(
        image,  # Input image
        number_of_times_to_upsample=upsample,  # Increase this value to detect smaller faces
        model=detector_model  # Use "hog" for speed or "cnn" for better accuracy
    )
    if not face_locations:  # Check whether any face was detected
        raise ValueError(f"No face detected in image: {image_path}")  # Raise error if no face is found
    selected_face = get_largest_face(face_locations)  # Use the largest face when multiple faces exist
    encodings = face_recognition.face_encodings(
        image,  # Input image
        known_face_locations=[selected_face],  # Extract encoding from selected face only
        num_jitters=num_jitters,  # More jitters may improve stability but reduce speed
        model=encoding_model  # Use "small" for speed or "large" for better accuracy
    )
    if not encodings:  # Check whether face encoding was generated
        raise ValueError(f"Failed to extract face encoding: {image_path}")  # Raise error if encoding fails
    return encodings[0]  # Return the 128-dimensional face embedding


def compare_face_similarity(
    image_path_1,
    image_path_2,
    threshold=0.93,
    detector_model="cnn",
    encoding_model="large",
    upsample=1,
    num_jitters=1
):
    encoding_1 = get_face_encoding(
        image_path_1,  # First image path
        detector_model=detector_model,  # Face detection model
        encoding_model=encoding_model,  # Face encoding model
        upsample=upsample,  # Face detection upsample count
        num_jitters=num_jitters  # Face encoding jitter count
    )
    encoding_2 = get_face_encoding(
        image_path_2,  # Second image path
        detector_model=detector_model,  # Face detection model
        encoding_model=encoding_model,  # Face encoding model
        upsample=upsample,  # Face detection upsample count
        num_jitters=num_jitters  # Face encoding jitter count
    )
    cos_sim = cosine_similarity(encoding_1, encoding_2)  # Calculate cosine similarity
    distance = float(np.linalg.norm(encoding_1 - encoding_2))  # Calculate Euclidean distance
    return {
        "is_same_person": cos_sim >= threshold,  # True if similarity is greater than or equal to threshold
        "cosine_similarity": round(cos_sim, 6),  # Cosine similarity value
        "cosine_percent": round(cos_sim * 100, 2),  # Cosine similarity percentage
        "euclidean_distance": round(distance, 6),  # Euclidean distance, lower means more similar
        "threshold": threshold  # Similarity threshold used for comparison
    }
if __name__ == "__main__":  # Run this block only when the file is executed directly
    result = compare_face_similarity(
        "person1.jpg",  # First face image
        "person2.jpg",  # Second face image
        threshold=0.93,  # Cosine similarity threshold
        detector_model="cnn",  # Face detector: "cnn" is more accurate, "hog" is faster
        encoding_model="large",  # Face encoding model: "large" is more accurate, "small" is faster
        upsample=1,  # Increase to detect smaller faces
        num_jitters=1  # Increase for more stable encoding but slower speed
    )

    print(result)  # Print comparison result
