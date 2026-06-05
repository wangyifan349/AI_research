from deepface import DeepFace

result = DeepFace.verify(
    img1_path="person1.jpg",        # First image
    img2_path="person2.jpg",        # Second image
    model_name="ArcFace",           # High-accuracy face recognition model
    detector_backend="retinaface",  # Accurate face detector
    distance_metric="cosine",       # Lower distance means more similar
    align=True,                     # Align faces for better accuracy
    enforce_detection=True,         # Raise error if no face is detected
)

print("Image 1:", "person1.jpg")
print("Image 2:", "person2.jpg")
print("Model:", result["model"])
print("Detector:", "retinaface")
print("Distance metric:", result["distance_metric"])
print(f"Cosine distance: {result['distance']:.6f}")      # Lower is better
print(f"Cosine similarity: {1 - result['distance']:.6f}") # Higher is better
print(f"Threshold: {result['threshold']:.6f}")
print("Verified:", result["verified"])
print("Decision:", "MATCH" if result["verified"] else "NOT MATCH")
