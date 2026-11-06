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
    spectrum = np.fft.fft2(image)
    spectrum_shifted = np.fft.fftshift(spectrum)
    magnitude = np.abs(spectrum_shifted)
    phase = np.angle(spectrum_shifted)
    return spectrum_shifted, magnitude, phase


def reconstruct_from_mag_phase(magnitude, phase):
    """Reconstruct a spatial image from shifted magnitude and phase."""
    spectrum_shifted = magnitude * np.exp(1j * phase)
    spectrum = np.fft.ifftshift(spectrum_shifted)
    image = np.fft.ifft2(spectrum)
    return np.real(image)


def compute_mad(a, b):
    """Compute the mean absolute difference before any display clipping."""
    return np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)))


def box_kernel(size=9):
    """Return a normalized box kernel of shape (size, size) in float64."""
    kernel = np.ones((size, size), dtype=np.float64)
    return kernel / kernel.sum()


def gaussian_kernel(size=9, sigma=1.5):
    """Return a normalized 2D Gaussian kernel in float64."""
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def spatial_convolution(image, kernel):
    """Apply 2D linear convolution with zero padding and the same output size.

    Implement the convolution without using cv2.filter2D or cv2.GaussianBlur.
    Remember to reverse the kernel to perform convolution rather than correlation.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    flipped_kernel = kernel[::-1, ::-1] 

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    h, w = image.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * flipped_kernel)
    return out


def fft_convolution(image, kernel):
    """Apply 2D linear convolution via numpy.fft with the same output size.

    Hint: pad for full convolution, then crop around the kernel center.
    Use the same zero-padding boundary convention as spatial_convolution.
    """
    h, w = image.shape
    kh, kw = kernel.shape
    full_h, full_w = h + kh - 1, w + kw - 1

    image_padded = np.zeros((full_h, full_w), dtype=np.float64)
    image_padded[:h, :w] = image

    kernel_padded = np.zeros((full_h, full_w), dtype=np.float64)
    kernel_padded[:kh, :kw] = kernel

    spectrum_product = np.fft.fft2(image_padded) * np.fft.fft2(kernel_padded)
    full_result = np.real(np.fft.ifft2(spectrum_product))

    pad_h, pad_w = kh // 2, kw // 2
    return full_result[pad_h:pad_h + h, pad_w:pad_w + w]


def part_a():
    """Compare magnitude-phase swaps, report four MAD values, and save figures."""
    a = load_grayscale("1.png")
    b = load_grayscale("2.png")
    if a.shape != b.shape:
        raise ValueError("1.png and 2.png must have the same shape")
    
    _, a_madnitudes, a_phase = compute_fft(a)
    _, b_madnitudes, b_phase = compute_fft(b)
    
    reconstructed1 = reconstruct_from_mag_phase(a_madnitudes, b_phase)
    reconstructed2 = reconstruct_from_mag_phase(b_madnitudes, a_phase)
    
    print("MAD(mag1+phase2, img1):", compute_mad(reconstructed1, a))
    print("MAD(mag1+phase2, img2):", compute_mad(reconstructed1, b))
    print("MAD(mag2+phase1, img1):", compute_mad(reconstructed2, a))
    print("MAD(mag2+phase1, img2):", compute_mad(reconstructed2, b))
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax in axes.ravel():
        ax.axis('off')

    axes[0, 0].imshow(a, cmap='gray'); axes[0, 0].set_title("Image 1")
    axes[0, 1].imshow(np.log1p(a_madnitudes), cmap='gray'); axes[0, 1].set_title("Log-magnitude 1")
    axes[0, 2].imshow(a_phase, cmap='gray'); axes[0, 2].set_title("Phase 1")

    axes[1, 0].imshow(b, cmap='gray'); axes[1, 0].set_title("Image 2")
    axes[1, 1].imshow(np.log1p(b_madnitudes), cmap='gray'); axes[1, 1].set_title("Log-magnitude 2")
    axes[1, 2].imshow(b_phase, cmap='gray'); axes[1, 2].set_title("Phase 2")

    axes[2, 0].imshow(reconstructed1, cmap='gray'); axes[2, 0].set_title("Mag(1) + Phase(2)")
    axes[2, 1].imshow(reconstructed2, cmap='gray'); axes[2, 1].set_title("Mag(2) + Phase(1)")

    plt.tight_layout()
    fig.savefig(OUTPUT / "part_a_magnitude_phase.png")
    plt.show()
    
    cv2.imwrite(str(OUTPUT / "recon_mag1_phase2.png"), np.clip(reconstructed1, 0, 255).astype(np.uint8))
    cv2.imwrite(str(OUTPUT / "recon_mag2_phase1.png"), np.clip(reconstructed2, 0, 255).astype(np.uint8))


def part_b():
    """Compare spatial-domain and frequency-domain filtering on kitty.png."""
    image = load_grayscale("kitty.png")

    box = box_kernel(9)
    gauss = gaussian_kernel(9, 1.5)

    box_spatial = spatial_convolution(image, box)
    box_freq = fft_convolution(image, box)
    gauss_spatial = spatial_convolution(image, gauss)
    gauss_freq = fft_convolution(image, gauss)

    mad_box = compute_mad(box_spatial, box_freq)
    mad_gauss = compute_mad(gauss_spatial, gauss_freq)
    print(f"MAD box (spatial vs frequency): {mad_box:.2e}")
    print(f"MAD gaussian (spatial vs frequency): {mad_gauss:.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax in axes.ravel():
        ax.axis('off')
    axes[0, 0].imshow(image, cmap='gray'); axes[0, 0].set_title("Original")
    axes[0, 1].imshow(box_spatial, cmap='gray'); axes[0, 1].set_title("Box - spatial")
    axes[0, 2].imshow(box_freq, cmap='gray'); axes[0, 2].set_title("Box - frequency")
    axes[1, 0].imshow(image, cmap='gray'); axes[1, 0].set_title("Original")
    axes[1, 1].imshow(gauss_spatial, cmap='gray'); axes[1, 1].set_title("Gaussian - spatial")
    axes[1, 2].imshow(gauss_freq, cmap='gray'); axes[1, 2].set_title("Gaussian - frequency")
    plt.tight_layout()
    fig.savefig(OUTPUT / "part_b_filtering.png")
    plt.show()


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    part_a()
    part_b()
