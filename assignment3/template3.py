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
    square = 8
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    pattern = (((xx // square) + (yy // square)) % 2) * 255
    return pattern.astype(np.float64)


def gaussian_kernel(size=5, sigma=1.0):
    """Return a normalized 2D Gaussian kernel."""
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_smooth(image, kernel):
    """Apply Gaussian smoothing in float64 precision."""
    # Boundary convention: REFLECT padding.
    # Zero-padding would darken the borders more and more with every pyramid level
    # (repeated smoothing+downsampling compounds the artificial black edges),
    # so reflect padding is used here instead of Exercise 2's zero-padding.
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    h, w = image.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * kernel)
    return out


def gaussian_pyramid(image, levels=4, prefilter=True):
    """Downsample by two repeatedly, with or without Gaussian smoothing."""
    kernel = gaussian_kernel(size=5, sigma=1.0)
    pyramid = [image]
    current = image
    for _ in range(levels - 1):
        if prefilter:
            current = gaussian_smooth(current, kernel)
        current = current[::2, ::2]
        pyramid.append(current)
    return pyramid


def expand_to(image, target_shape):
    """Upsample to (height, width) using a documented interpolation mode."""
    # Interpolation: bilinear (cv2.INTER_LINEAR). Resizing directly to the
    # exact target_shape (rather than a fixed x2) handles odd dimensions
    # correctly, since each pyramid level's size isn't always exactly half
    # the level below when the original isn't a power of two.
    target_h, target_w = target_shape
    return cv2.resize(image.astype(np.float64), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def laplacian_pyramid(gaussian_levels):
    """L[l] = G[l] - expand(G[l+1]), and L[3] = G[3]."""
    laplacian_levels = []
    for l in range(len(gaussian_levels) - 1):
        expanded = expand_to(gaussian_levels[l + 1], gaussian_levels[l].shape)
        laplacian_levels.append(gaussian_levels[l] - expanded)
    laplacian_levels.append(gaussian_levels[-1])
    return laplacian_levels


def reconstruct(laplacian_levels):
    """Reconstruct the image by expanding and adding from coarse to fine."""
    current = laplacian_levels[-1]
    for l in range(len(laplacian_levels) - 2, -1, -1):
        expanded = expand_to(current, laplacian_levels[l].shape)
        current = laplacian_levels[l] + expanded
    return current


def compute_mad(a, b):
    return np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kitty = load_grayscale("kitty.png")
    # Either supplied pattern OR your own synthetic one is acceptable.
    pattern_file = next((p for p in ("chequerboard.jpg", "stripes.png")
                         if (DATA / p).exists()), None)
    pattern = load_grayscale(pattern_file) if pattern_file else make_high_frequency_pattern()
    
    # a
    for name, image in [("kitty", kitty), ("pattern", pattern)]:
        pyr_unfiltered = gaussian_pyramid(image, levels=4, prefilter=False)
        pyr_filtered = gaussian_pyramid(image, levels=4, prefilter=True)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for lvl in range(4):
            axes[0, lvl].imshow(pyr_unfiltered[lvl], cmap='gray')
            axes[0, lvl].set_title(f"{name} unfiltered L{lvl} {pyr_unfiltered[lvl].shape}")
            axes[0, lvl].axis('off')
            axes[1, lvl].imshow(pyr_filtered[lvl], cmap='gray')
            axes[1, lvl].set_title(f"{name} filtered L{lvl} {pyr_filtered[lvl].shape}")
            axes[1, lvl].axis('off')
        plt.tight_layout()
        fig.savefig(OUTPUT / f"gaussian_pyramid_{name}.png")
        plt.show()

    # b
    kitty_gaussian = gaussian_pyramid(kitty, levels=4, prefilter=True)
    kitty_laplacian = laplacian_pyramid(kitty_gaussian)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for lvl, lap in enumerate(kitty_laplacian):
        axes[lvl].imshow(lap, cmap='gray')
        axes[lvl].set_title(f"Laplacian L{lvl} {lap.shape}")
        axes[lvl].axis('off')
    plt.tight_layout()
    fig.savefig(OUTPUT / "laplacian_pyramid_kitty.png")
    plt.show()

    reconstructed = reconstruct(kitty_laplacian)
    mad = compute_mad(reconstructed, kitty)
    print(f"Reconstruction MAD: {mad:.2e}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(kitty, cmap='gray'); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(reconstructed, cmap='gray'); axes[1].set_title("Reconstructed"); axes[1].axis('off')
    plt.tight_layout()
    fig.savefig(OUTPUT / "reconstruction.png")
    plt.show()


if __name__ == "__main__":
    main()
