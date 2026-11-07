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


def convolve2d_same(image, kernel, pad_mode="reflect"):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)
    h, w = image.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * kernel)
    return out

def gaussian_smoothing(image, size=5, sigma=1.0):
    """Apply Gaussian smoothing without using cv2.GaussianBlur."""
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    smoothed = convolve2d_same(image, kernel, pad_mode="reflect")
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def sobel_gradients(image):
    """Compute horizontal and vertical image gradients."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    dx = convolve2d_same(image, sobel_x, pad_mode="reflect")
    dy = convolve2d_same(image, sobel_y, pad_mode="reflect")
    return dx, dy


def non_maximum_suppression(magnitude, dx, dy):
    """Thin magnitude ridges along the gradient direction."""
    h, w = magnitude.shape
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    thin = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            if a < 22.5 or a >= 157.5:
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]            # 0 deg
            elif a < 67.5:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]    # 45 deg
            elif a < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]            # 90 deg
            else:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]    # 135 deg

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                thin[i, j] = magnitude[i, j]
    return thin


def hysteresis(thin, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    """Keep weak edges only if 8-connected to a strong edge."""
    h, w = thin.shape
    strong = thin >= high
    weak = (thin >= low) & (thin < high)
     
    result = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    stack = list(zip(*np.nonzero(strong)))
    for i, j in stack:
        result[i, j] = 255
        visited[i, j] = True
          
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while stack:
        i, j = stack.pop()
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and weak[ni, nj]:
                result[ni, nj] = 255
                visited[ni, nj] = True  
                stack.append((ni, nj))
    return result


def my_canny(image, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    """Compute Sobel gradients, gradient magnitude, NMS, and hysteresis."""
    smoothed = gaussian_smoothing(image, size=5, sigma=1.0)
    dx, dy = sobel_gradients(smoothed)
    magnitude = np.sqrt(dx**2 + dy**2)
    thin = non_maximum_suppression(magnitude, dx, dy)
    return hysteresis(thin, low=low, high=high)


def compute_metrics(custom, reference):
    """Compute F1-score and MAD between two binary edge maps."""
    custom_bin = (custom > 0).astype(np.uint8)
    reference_bin = (reference > 0).astype(np.uint8)

    mad = np.mean(np.abs(custom_bin.astype(np.float64) - reference_bin.astype(np.float64)))

    tp = np.sum((custom_bin == 1) & (reference_bin == 1))
    fp = np.sum((custom_bin == 1) & (reference_bin == 0))
    fn = np.sum((custom_bin == 0) & (reference_bin == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, mad


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = load_grayscale("uni-bonn.jpg")

    smoothed = gaussian_smoothing(image, size=5, sigma=1.0)

    custom_edges = my_canny(smoothed, low=LOW_THRESHOLD, high=HIGH_THRESHOLD)

    reference_edges = cv2.Canny(smoothed, LOW_THRESHOLD, HIGH_THRESHOLD,apertureSize=3, L2gradient=True)

    f1, mad = compute_metrics(custom_edges, reference_edges)
    print(f"F1-score: {f1:.4f}")
    print(f"MAD: {mad:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(image, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(custom_edges, cmap='gray'); axes[1].set_title("My Canny")
    axes[2].imshow(reference_edges, cmap='gray'); axes[2].set_title("cv2.Canny")
    plt.tight_layout()
    fig.savefig(OUTPUT / "canny_comparison.png")
    plt.show()

    cv2.imwrite(str(OUTPUT / "my_canny.png"), custom_edges)
    cv2.imwrite(str(OUTPUT / "cv2_canny.png"), reference_edges)


if __name__ == "__main__":
    main()
