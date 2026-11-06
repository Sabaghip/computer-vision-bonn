"""
Exercise Sheet 2 - Exercise 2
MA-INF 2201 Computer Vision WS26/27
Fourier Transform and Spatial vs. Frequency Filtering - Template
Python 3.12, OpenCV 5.0.0, NumPy 2.3.3
Images: data/1.png, data/2.png, data/kitty.png

Do not use cv2.filter2D or cv2.GaussianBlur for spatial filtering.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "exercise2"


def load_grayscale(filename):
    """Load an image as float64, preserving grayscale intensities."""
    image = cv2.imread(str(DATA / filename), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing input image: {DATA / filename}")
    return image.astype(np.float64)


def compute_fft(image):
    """Return shifted complex spectrum, magnitude, and phase."""
    # TODO
    raise NotImplementedError


def reconstruct_from_mag_phase(magnitude, phase):
    """Reconstruct a spatial image from shifted magnitude and phase."""
    # TODO: Reverse the frequency shift and apply the inverse 2D DFT.
    raise NotImplementedError


def compute_mad(a, b):
    """Compute the mean absolute difference before any display clipping."""
    # TODO
    raise NotImplementedError


def box_kernel(size=9):
    """Return a normalized box kernel of shape (size, size) in float64."""
    # TODO
    raise NotImplementedError


def gaussian_kernel(size=9, sigma=1.5):
    """Return a normalized 2D Gaussian kernel in float64."""
    # TODO
    raise NotImplementedError


def spatial_convolution(image, kernel):
    """Apply 2D linear convolution with zero padding and the same output size.

    Implement the convolution without using cv2.filter2D or cv2.GaussianBlur.
    Remember to reverse the kernel to perform convolution rather than correlation.
    """
    # TODO
    raise NotImplementedError


def fft_convolution(image, kernel):
    """Apply 2D linear convolution via numpy.fft with the same output size.

    Hint: pad for full convolution, then crop around the kernel center.
    Use the same zero-padding boundary convention as spatial_convolution.
    """
    # TODO
    raise NotImplementedError


def part_a():
    """Compare magnitude-phase swaps, report four MAD values, and save figures."""
    a = load_grayscale("1.png")
    b = load_grayscale("2.png")
    if a.shape != b.shape:
        raise ValueError("1.png and 2.png must have the same shape")
    # TODO: Compute spectra, magnitudes, phases of both images.
    # TODO: Reconstruct images using magnitude(1) + phase(2) and vice versa.
    # TODO: Report the MAD of each reconstruction against both originals.
    # TODO: Visualize the originals, log-magnitudes, phases, and reconstructions.
    # TODO: Save the reconstructed images and figures in OUTPUT.
    raise NotImplementedError


def part_b():
    """Compare spatial-domain and frequency-domain filtering on kitty.png."""
    image = load_grayscale("kitty.png")
    # TODO: Create normalized 9x9 box and Gaussian kernels (sigma=1.5).
    # TODO: Apply both kernels using spatial_convolution and fft_convolution.
    # TODO: Visualize the results and report both MAD values (< 1e-7).
    # TODO: Save the figures in OUTPUT.
    raise NotImplementedError


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    part_a()
    part_b()
