"""
pip install insightface onnxruntime opencv-python numpy huggingface_hub

This script detects faces, estimates age and gender, extracts face embeddings, and compares two faces using cosine similarity.

"""

import cv2                                      # Image loading and basic image processing
import numpy as np                             # Vector similarity calculation
from huggingface_hub import snapshot_download  # Download models from Hugging Face
from insightface.app import FaceAnalysis       # Main InsightFace face analysis API


snapshot_download(                             # Download the buffalo_l model package
    repo_id="lithiumice/insightface",          # Hugging Face repository
    allow_patterns=["models/buffalo_l/*.onnx"],# Download only ONNX files under buffalo_l
    local_dir="./hf_models",                   # Local model directory
)

app = FaceAnalysis(                            # Initialize face analysis app
    name="buffalo_l",                          # Use the buffalo_l model pack
    root="./hf_models",                        # Model root directory
    providers=["CPUExecutionProvider"],        # Use CPU inference
)

app.prepare(                                   # Prepare models
    ctx_id=-1,                                 # -1 means CPU; 0 means GPU device 0
    det_size=(640, 640),                       # Face detection input size
)


# 1. Single image analysis: age, gender, face box, landmarks
img = cv2.imread("test.jpg")                   # Read the image to analyze
faces = app.get(img)                           # Detect faces and extract attributes

for i, face in enumerate(faces):               # Iterate over all detected faces
    gender = (                                 # Convert gender label to text
        "female" if face.gender == 0
        else "male" if face.gender == 1
        else "unknown"
    )

    print(f"\nFace {i}")                       # Print face index
    print("bbox:", face.bbox.astype(int).tolist())          # Face box: [x1, y1, x2, y2]
    print("score:", float(face.det_score))                  # Face detection confidence
    print("age:", int(face.age))                            # Estimated age
    print("gender:", gender)                                # Estimated gender
    print("landmarks:", face.kps.astype(int).tolist())      # 5-point facial landmarks
    print("embedding_dim:", face.normed_embedding.shape[0]) # Face embedding dimension, usually 512


# 2. Face comparison: a.jpg vs b.jpg
img1 = cv2.imread("a.jpg")                     # Read the first comparison image
img2 = cv2.imread("b.jpg")                     # Read the second comparison image

faces1 = app.get(img1)                         # Detect faces in the first image
faces2 = app.get(img2)                         # Detect faces in the second image

if not faces1 or not faces2:                   # If either image has no detected face
    raise RuntimeError("No face detected in a.jpg or b.jpg")

face1 = faces1[0]                              # If multiple faces exist, use the first one
face2 = faces2[0]                              # If multiple faces exist, use the first one

similarity = float(                            # Compute cosine similarity
    np.dot(face1.normed_embedding, face2.normed_embedding)
)
threshold = 0.45                               # Decision threshold; tune it for your data

print("\nCompare")                             # Print comparison section title
print("similarity:", similarity)               # Similarity score; higher means more similar
print("same_person:", similarity >= threshold) # Whether they are classified as the same person
