#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_match_all.py – Unified demo for SIFT / SURF / ORB / AKAZE / KAZE / BRISK
Saves a visualised match image and (when possible) the homography matrix
for each algorithm.
"""

import cv2
import numpy as np
from pathlib import Path

# ----------- Configuration -----------
IMG1 = "query.jpg"          # path to the query image
IMG2 = "train.jpg"          # path to the train image
OUT_DIR = Path("matches_out")
OUT_DIR.mkdir(exist_ok=True)

# Algorithms to demonstrate
ALGORITHMS = [
    "SIFT",
    "SURF",
    "ORB",
    "AKAZE",
    "KAZE",
    "BRISK",
]

# ----------- Helper functions -----------

def create_detector(method: str):
    """Return an OpenCV key-point detector / descriptor extractor by name."""
    if method == "SIFT":
        return cv2.SIFT_create()
    if method == "SURF":
        # Higher hessianThreshold → fewer key-points detected
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    if method == "ORB":
        return cv2.ORB_create(nfeatures=2000)
    if method == "AKAZE":
        return cv2.AKAZE_create()
    if method == "KAZE":
        return cv2.KAZE_create()
    if method == "BRISK":
        return cv2.BRISK_create()
    raise ValueError(f"Unknown method: {method}")

def create_matcher(method: str):
    """Return a matcher suited to the descriptor type of the given algorithm."""
    if method in {"SIFT", "SURF", "KAZE"}:            # float descriptors → L2 distance
        index_params = dict(algorithm=1, trees=5)     # 1 = KD-Tree
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)
    # binary descriptors → Hamming distance
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def ratio_test(matches, ratio: float = 0.75):
    """Apply Lowe’s ratio test to K-NN matches."""
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def match_and_draw(method: str) -> None:
    print(f"\n=== {method} ===")
    detector = create_detector(method)
    matcher  = create_matcher(method)

    # 1. Read images & extract features
    img1 = cv2.imread(IMG1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(IMG2, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # 2. Descriptor matching
    if isinstance(matcher, cv2.FlannBasedMatcher):
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good = ratio_test(
            raw_matches,
            ratio=0.75 if method in {"SIFT", "SURF"} else 0.80,
        )
    else:
        raw_matches = matcher.match(des1, des2)
        good = sorted(raw_matches, key=lambda m: m.distance)[: int(len(raw_matches) * 0.8)]

    print(
        f"Key-points: {len(kp1)} vs {len(kp2)} | "
        f"Good matches after filtering: {len(good)}"
    )

    # 3. Homography estimation
    H, mask = None, None
    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print("Homography matrix H =\n", H)
    else:
        print("Not enough matches to compute a homography.")

    # 4. Draw matches (inliers only if a mask is available)
    matches_mask = mask.ravel().tolist() if mask is not None else None
    vis = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    out_path = OUT_DIR / f"{method.lower()}_matches.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Visualisation saved → {out_path}")

# ----------- Main routine -----------
def main() -> None:
    for algo in ALGORITHMS:
        match_and_draw(algo)

if __name__ == "__main__":
    main()
