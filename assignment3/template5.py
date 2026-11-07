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
    h, w = edge_mask.shape
    dist = np.where(edge_mask > 0, 0.0, np.inf)
     

    forward_offsets = [
        (-1, -1, 7), (-1, 0, 5), (-1, 1, 7), (0, -1, 5),
        (-2, -1, 11), (-2, 1, 11), (-1, -2, 11), (-1, 2, 11),
    ]
    for i in range(h):
        for j in range(w):
            best = dist[i, j]
            for di, dj, cost in forward_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    candidate = dist[ni, nj] + cost
                    if candidate < best:
                        best = candidate
            dist[i, j] = best

    backward_offsets = [(-di, -dj, cost) for di, dj, cost in forward_offsets]
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            best = dist[i, j]
            for di, dj, cost in backward_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    candidate = dist[ni, nj] + cost
                    if candidate < best:
                        best = candidate
            dist[i, j] = best
    return dist


def directional_chamfer(a, b):
    """Return the average distance from A's edge pixels to the nearest edge in B.

    Express the result in approximate pixel units (raw Chamfer / 5).
    """
    dist_to_b = chamfer_5711(b)
    return np.mean(dist_to_b[a > 0]) / 5.0


def make_synthetic_edge_maps():
    """Return two binary edge maps, with B strictly containing A's edge pixels."""
    size = 10
    A = np.zeros((size, size), dtype=np.uint8)
    A[3, 1:3] = 1
    A[4:5, 6] = 1 

    B = A.copy()
    B[8, 1] = 1 

    return A, B


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = load_grayscale("uni-bonn.jpg")
    edges = cv2.Canny(image, 100, 200)

    chamfer_raw = chamfer_5711(edges)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(image, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(edges, cmap='gray'); axes[1].set_title("Canny edges")
    im = axes[2].imshow(chamfer_raw, cmap='jet'); axes[2].set_title("Chamfer 5-7-11 distance")
    plt.tight_layout()
    fig.savefig(OUTPUT / "chamfer_distance.png")
    plt.show()

    inverted_mask = np.where(edges > 0, 0, 255).astype(np.uint8)
    euclidean_dist = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    chamfer_pixel_units = chamfer_raw / 5.0

    diff = np.abs(chamfer_pixel_units - euclidean_dist)
    print(f"Mean abs diff (chamfer/5 vs. Euclidean): {diff.mean():.4f}")
    print(f"Max abs diff (chamfer/5 vs. Euclidean): {diff.max():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(chamfer_pixel_units, cmap='jet'); axes[0].set_title("Chamfer / 5")
    axes[1].imshow(euclidean_dist, cmap='jet'); axes[1].set_title("cv2 Euclidean DT")
    axes[2].imshow(diff, cmap='jet'); axes[2].set_title("Absolute difference")
    plt.tight_layout()
    fig.savefig(OUTPUT / "chamfer_vs_euclidean.png")
    plt.show()

    A, B = make_synthetic_edge_maps()
    d_ab = directional_chamfer(A, B)
    d_ba = directional_chamfer(B, A)
    print(f"d_ch(A, B) = {d_ab:.4f}")
    print(f"d_ch(B, A) = {d_ba:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(A, cmap='gray'); axes[0].set_title("A"); axes[0].axis('off')
    axes[1].imshow(B, cmap='gray'); axes[1].set_title("B"); axes[1].axis('off')
    plt.tight_layout()
    fig.savefig(OUTPUT / "synthetic_edge_maps.png")
    plt.show()


if __name__ == "__main__":
    main()
