"""
Exercise Sheet 2 - Exercise 5
MA-INF 2201 Computer Vision WS26/27
Chamfer Distance - Template
Python 3.12, OpenCV 5.0.0, NumPy 2.3.3
Image: data/uni-bonn.jpg

Using cv2.Canny for edge detection is permitted.
Implement the Chamfer distance transform using two raster scans.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "exercise5"


def load_grayscale(filename):
    image = cv2.imread(str(DATA / filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing input image: {DATA / filename}")
    return image


def chamfer_5711(edge_mask):
    """Return the Chamfer distance transform in raw 5-7-11 units.

    edge_mask: Nonzero pixels represent edge pixels (distance zero).
    Costs: (0, 1)/(1, 0) -> 5, (1, 1) -> 7,
           (1, 2)/(2, 1) -> 11, with sign changes.
    Use forward and backward raster scans.
    Do not use built-in distance transform functions.
    """
    # TODO: Initialize distances to zero at edge pixels and infinity elsewhere.
    # TODO: Implement the predecessor offsets for the forward pass.
    # TODO: Implement the corresponding offsets for the backward pass.
    raise NotImplementedError


def directional_chamfer(a, b):
    """Return the average distance from A's edge pixels to the nearest edge in B.

    Express the result in approximate pixel units (raw Chamfer / 5).
    """
    # TODO: Compute chamfer_5711(b) and average the distances at A's edge pixels.
    raise NotImplementedError


def make_synthetic_edge_maps():
    """Return two binary edge maps, with B strictly containing A's edge pixels."""
    # TODO: Create two small arrays with the same shape.
    raise NotImplementedError


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = load_grayscale("uni-bonn.jpg")
    edges = cv2.Canny(image, 100, 200)
    # TODO: Compute the Chamfer 5-7-11 transform.
    # TODO: Visualize the input image, edge map, and distance transform.
    # TODO: Invert the edge mask for cv2.distanceTransform (zeros at edge pixels).
    # TODO: Compare Chamfer / 5 with DIST_L2 and visualize the absolute difference.
    # TODO: Compute d_ch(A, B) and d_ch(B, A) and explain the asymmetry.
    raise NotImplementedError


if __name__ == "__main__":
    main()
