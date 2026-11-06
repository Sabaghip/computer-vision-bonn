"""
Exercise Sheet 2 - Exercise 4
MA-INF 2201 Computer Vision WS26/27
Canny Edge Detector - Template
Python 3.12, OpenCV 5.0.0, NumPy 2.3.3
Image: data/uni-bonn.jpg

Implement the Canny edge detector without using cv2.Canny.
Use cv2.Canny only as a reference for comparison.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "exercise4"
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 100


def load_grayscale(filename):
    image = cv2.imread(str(DATA / filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing input image: {DATA / filename}")
    return image


def gaussian_smoothing(image, size=5, sigma=1.0):
    """Apply Gaussian smoothing without using cv2.GaussianBlur."""
    # TODO: Return a uint8 image to ensure identical inputs for both Canny detectors.
    raise NotImplementedError


def sobel_gradients(image):
    """Compute horizontal and vertical image gradients."""
    # TODO: Return (dx, dy) using consistent border handling.
    raise NotImplementedError


def non_maximum_suppression(magnitude, dx, dy):
    """Thin magnitude ridges along the gradient direction."""
    # TODO: Compare neighbors along the gradient direction.
    raise NotImplementedError


def hysteresis(thin, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    """Keep weak edges only if 8-connected to a strong edge."""
    # TODO: Implement 8-neighbor propagation using a stack or queue.
    raise NotImplementedError


def my_canny(image, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    """Compute Sobel gradients, gradient magnitude, NMS, and hysteresis."""
    # TODO: Do not use cv2.Canny in this implementation.
    raise NotImplementedError


def compute_metrics(custom, reference):
    """Compute F1-score and MAD between two binary edge maps."""
    # TODO: Compute precision, recall, F1-score, and MAD using binary values (0, 1).
    raise NotImplementedError


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = load_grayscale("uni-bonn.jpg")
    # TODO: Apply Gaussian smoothing once and use the same preprocessed uint8 image
    #       for both your implementation and the cv2.Canny reference.
    # TODO: Use compatible Sobel aperture and L1/L2 gradient settings.
    # TODO: Compute MAD (<= 0.07) and F1-score (>= 0.6).
    # TODO: Save and visualize the resulting edge maps.
    raise NotImplementedError


if __name__ == "__main__":
    main()
