#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_matching_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~
A self-contained demonstration of six classic local-feature pipelines
available in OpenCV-contrib 4.4 +:

* SIFT  – Scale-Invariant Feature Transform
* SURF  – Speeded-Up Robust Features
* ORB   – Oriented FAST and Rotated BRIEF
* AKAZE – Accelerated KAZE
* KAZE  – Non-linear scale-space variant
* BRISK – Binary Robust Invariant Scalable Keypoints

For each algorithm the script

1. detects keypoints and extracts descriptors on the **query** and **train**
   images;
2. matches descriptors with the proper distance metric  
   (L2 for floating-point, Hamming for binary);
3. removes outliers with Lowe’s ratio test (threshold chosen per algorithm);
4. estimates a homography with RANSAC when ≥ 4 good matches are available;
5. draws inlier matches and writes «<algorithm>_matches.jpg» files into the
   output directory;
6. prints concise statistics (# keypoints, # good matches, inlier ratio,
   homography).

Dependencies
------------
Python 3.8 +, NumPy, and OpenCV («opencv-contrib-python» wheel)

    pip install opencv-contrib-python numpy
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# User-adjustable constants (edit here, no CLI needed)                        #
# --------------------------------------------------------------------------- #

QUERY_IMAGE_PATH: str = "query.jpg"          # path to the query image
TRAIN_IMAGE_PATH: str = "train.jpg"          # path to the train image
OUTPUT_DIRECTORY: Path = Path("matches_out")  # folder for visualisations

ALGORITHMS: list[str] = ["SIFT", "SURF", "ORB", "AKAZE", "KAZE", "BRISK"]

# Empirically chosen Lowe ratio thresholds per algorithm
LOWE_RATIO_THRESHOLD: dict[str, float] = {
    "SIFT": 0.75,
    "SURF": 0.75,
    "KAZE": 0.75,
    "ORB": 0.80,
    "AKAZE": 0.80,
    "BRISK": 0.80,
}
# --------------------------------------------------------------------------- #
# Factory helpers                                                             #
# --------------------------------------------------------------------------- #
def create_feature_detector(name: str) -> cv2.Feature2D:
    """Return an OpenCV feature detector / descriptor extractor."""
    if name == "SIFT":
        return cv2.SIFT_create()
    if name == "SURF":
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    if name == "ORB":
        return cv2.ORB_create(nfeatures=2_000)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    if name == "KAZE":
        return cv2.KAZE_create()
    if name == "BRISK":
        return cv2.BRISK_create()
    raise ValueError(f"Unknown algorithm: {name}")

def create_descriptor_matcher(name: str) -> cv2.DescriptorMatcher:
    """Return a matcher suited to the algorithm’s descriptor type."""
    if name in {"SIFT", "SURF", "KAZE"}:              # float descriptors
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    # binary descriptors
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def apply_lowes_ratio_test(
    knn_matches: list[list[cv2.DMatch]], ratio_threshold: float
) -> list[cv2.DMatch]:
    """Filter KNN matches using Lowe’s ratio test."""
    return [
        first for first, second in knn_matches
        if first.distance < ratio_threshold * second.distance
    ]
# --------------------------------------------------------------------------- #
# Core pipeline                                                               #
# --------------------------------------------------------------------------- #
def process_pipeline(
    algorithm: str,
    query_image: np.ndarray,
    train_image: np.ndarray,
    output_dir: Path,
) -> None:
    """Detect, match, estimate homography and visualise results."""
    detector = create_feature_detector(algorithm)
    matcher = create_descriptor_matcher(algorithm)
    ratio_threshold = LOWE_RATIO_THRESHOLD[algorithm]
    # 1  Detect & describe
    keypoints_query, descriptors_query = detector.detectAndCompute(query_image, None)
    keypoints_train, descriptors_train = detector.detectAndCompute(train_image, None)
    if descriptors_query is None or descriptors_train is None:
        print(f"[{algorithm}] No descriptors found – skipped.")
        return
    # 2  Match descriptors
    if isinstance(matcher, cv2.FlannBasedMatcher):
        knn_matches = matcher.knnMatch(descriptors_query, descriptors_train, k=2)
        good_matches = apply_lowes_ratio_test(knn_matches, ratio_threshold)
    else:
        raw_matches = matcher.match(descriptors_query, descriptors_train)
        raw_matches.sort(key=lambda m: m.distance)
        good_matches = raw_matches[: int(len(raw_matches) * 0.8)]
    # 3  Estimate homography
    homography_matrix, inlier_mask = None, None
    if len(good_matches) >= 4:
        src_points = np.float32(
            [keypoints_query[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_points = np.float32(
            [keypoints_train[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        homography_matrix, inlier_mask = cv2.findHomography(
            src_points, dst_points, cv2.RANSAC, 5.0
        )
    # 4  Visualise
    matches_mask = inlier_mask.ravel().tolist() if inlier_mask is not None else None
    visualisation = cv2.drawMatches(
        query_image,
        keypoints_query,
        train_image,
        keypoints_train,
        good_matches,
        None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output_path = output_dir / f"{algorithm.lower()}_matches.jpg"
    cv2.imwrite(str(output_path), visualisation)
    # 5  Statistics
    inliers = sum(matches_mask) if matches_mask else 0
    inlier_ratio = inliers / len(good_matches) if good_matches else 0.0
    print(
        f"[{algorithm}] keypoints: {len(keypoints_query):4d}/{len(keypoints_train):4d} | "
        f"good matches: {len(good_matches):4d} | "
        f"inliers: {inliers:4d} ({inlier_ratio:.1%})"
    )
    if homography_matrix is not None:
        print(f"[{algorithm}] Homography matrix:\n{homography_matrix}\n")
    else:
        print(f"[{algorithm}] Homography could not be estimated.\n")
# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    query_image = cv2.imread(QUERY_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    train_image = cv2.imread(TRAIN_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if query_image is None or train_image is None:
        sys.exit(
            "Error: unable to read the input images. "
            "Edit QUERY_IMAGE_PATH / TRAIN_IMAGE_PATH at the top of the script."
        )
    for algorithm in ALGORITHMS:
        process_pipeline(algorithm, query_image, train_image, OUTPUT_DIRECTORY)
if __name__ == "__main__":
    main()
