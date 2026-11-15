"""
Task 3: Hough Transform for Circle Detection.
Template for MA-INF 2201 Computer Vision WS26/27
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def hough_circles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    max_radius = min(max_radius, int(np.linalg.norm(edges.shape)))

    edges_points = np.array(np.nonzero(edges))
    h, w = edges.shape

    radii = np.arange(min_radius, max_radius + 1, r_ssz)
    thetas = np.deg2rad(np.arange(0, 360, theta_ssz))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    ys = edges_points[0].astype(np.float64)
    xs = edges_points[1].astype(np.float64)

    accumulator = np.zeros((len(radii), h, w), dtype=np.int32)

    for r_idx, r in enumerate(radii):
        a = xs[:, None] - r * cos_t[None, :]
        b = ys[:, None] - r * sin_t[None, :]

        a_idx = np.round(a).astype(np.int64)
        b_idx = np.round(b).astype(np.int64)

        valid = (a_idx >= 0) & (a_idx < w) & (b_idx >= 0) & (b_idx < h)
        np.add.at(accumulator[r_idx], (b_idx[valid], a_idx[valid]), 1)

    detected_circles = []
    acc_copy = accumulator.copy()
    yy, xx = np.ogrid[:h, :w]
    while True:
        r_idx, b, a = np.unravel_index(np.argmax(acc_copy), acc_copy.shape)
        votes = acc_copy[r_idx, b, a]
        if votes < threshold:
            break
        detected_circles.append((int(a), int(b), int(radii[r_idx]), int(votes)))

        dist_sq = (xx - a) ** 2 + (yy - b) ** 2
        acc_copy[:, dist_sq <= min_dist ** 2] = 0 

    return detected_circles, accumulator

def main():
    """Main function."""
    print("=" * 70)
    print("Task 3: Hough Transform for Circle Detection")
    print("=" * 70)

    img_path = 'data/coins.jpg'
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    print("\nDetecting circles...")
    min_radius = 15
    max_radius = 70
    threshold = 30
    min_dist = 20
    r_ssz = 2
    theta_ssz = 5

    detected_circles, accumulator = hough_circles(
        edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz
    )

    print(f"Detected {len(detected_circles)} circles.")
    for a, b, r, v in detected_circles:
        print(f"  center=({a},{b})  radius={r}  votes={v}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for a, b, r, v in detected_circles:
        ax.add_patch(Circle((a, b), r, color='lime', fill=False, linewidth=2))
        ax.plot(a, b, 'r+', markersize=8)
    ax.set_title(f"Detected circles ({len(detected_circles)})")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    radii = np.arange(min_radius, max_radius + 1, r_ssz)
    sample_idxs = np.linspace(0, len(radii) - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, idx in zip(axes.ravel(), sample_idxs):
        ax.imshow(accumulator[idx], cmap='hot')
        ax.set_title(f"r={radii[idx]}")
        ax.axis('off')
    plt.suptitle("Accumulator slices at different radii")
    plt.tight_layout()
    plt.show()

    best_r_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)[0]
    plt.figure(figsize=(6, 5))
    plt.imshow(accumulator[best_r_idx], cmap='hot')
    plt.title(f"Best slice (r={radii[best_r_idx]}, max votes={accumulator[best_r_idx].max()})")
    plt.colorbar(label="votes")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print(f"min_radius={min_radius}, max_radius={max_radius}, threshold={threshold}, "
          f"min_dist={min_dist}, r_ssz={r_ssz}, theta_ssz={theta_ssz}")


if __name__ == "__main__":
    main()