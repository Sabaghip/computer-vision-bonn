"""
Exercise Sheet 2 - Exercise 3
MA-INF 2201 Computer Vision WS26/27
Gaussian and Laplacian Pyramids - Template
Python 3.12, OpenCV 5.0.0, NumPy 2.3.3
Image: data/kitty.png; optionally data/chequerboard.jpg, data/stripes.png

Construct Gaussian and Laplacian pyramids with four levels (0-3).
Generate a synthetic checkerboard if neither optional image is available.
Do not use cv2.pyrDown or cv2.pyrUp; cv2.resize is allowed for expansion.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "exercise3"


def load_grayscale(filename):
    image = cv2.imread(str(DATA / filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing input image: {DATA / filename}")
    return image.astype(np.float64)


def make_high_frequency_pattern(size=256):
    """Generate a synthetic high-frequency pattern with values 0 and 255."""
    # TODO
    raise NotImplementedError


def gaussian_kernel(size=5, sigma=1.0):
    """Return a normalized 2D Gaussian kernel."""
    # TODO
    raise NotImplementedError


def gaussian_smooth(image, kernel):
    """Apply Gaussian smoothing in float64 precision."""
    # TODO: You may reuse your spatial Gaussian filter from Exercise 2.
    # TODO: Choose and document a boundary convention.
    raise NotImplementedError


def gaussian_pyramid(image, levels=4, prefilter=True):
    """Downsample by two repeatedly, with or without Gaussian smoothing."""
    # TODO: Level 0 is the input; return a list of exactly `levels` images.
    raise NotImplementedError


def expand_to(image, target_shape):
    """Upsample to (height, width) using a documented interpolation mode."""
    # TODO: Use cv2.resize and handle odd image dimensions correctly.
    raise NotImplementedError


def laplacian_pyramid(gaussian_levels):
    """L[l] = G[l] - expand(G[l+1]), and L[3] = G[3]."""
    # TODO
    raise NotImplementedError


def reconstruct(laplacian_levels):
    """Reconstruct the image by expanding and adding from coarse to fine."""
    # TODO: Do not clip/round intermediate results.
    raise NotImplementedError


def compute_mad(a, b):
    # TODO
    raise NotImplementedError


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kitty = load_grayscale("kitty.png")
    # Either supplied pattern OR your own synthetic one is acceptable.
    pattern_file = next((p for p in ("chequerboard.jpg", "stripes.png")
                         if (DATA / p).exists()), None)
    pattern = load_grayscale(pattern_file) if pattern_file else make_high_frequency_pattern()
    # TODO: Build and visualize filtered and unfiltered Gaussian pyramids
    #       for both the natural image and the high-frequency pattern.
    # TODO: Build and visualize Laplacian levels from the filtered kitty pyramid.
    # TODO: Reconstruct, visualize, and print reconstruction MAD.
    # TODO: Explain aliasing and report Gaussian filter and interpolation settings.
    raise NotImplementedError


if __name__ == "__main__":
    main()
