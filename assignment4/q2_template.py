"""
Task 2: Roll correction with Hough Transform.
Template for MA-INF 2201 Computer Vision WS26/27
Exercise 03
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from q1_template import generate_accumulator

# Constants
# Do not change these
THETA_MIN = 75
THETA_MAX = 105

def correct_roll(
        accumulator: np.ndarray,
        theta_deg: np.ndarray,
        nd_rho: np.ndarray
    ) -> tuple[list[tuple[float, float]], float]:
    """Roll correction using topk dominant lines.

    Assume we consider horizontal lines and roll angle is within 15 degrees.
    Args:
        accumulator (np.ndarray): Accumulator cells.
        theta_deg (np.ndarray): Range of thetas in degrees.
        nd_rho (np.ndarray): Range of rhos.

    Returns:
        tuple[list[tuple[float, float]], float]: Tuple containing topk lines and correction angle.
    """
    flat_order = np.argsort(accumulator.ravel())[::-1]
    rho_idx_all, theta_idx_all = np.unravel_index(flat_order, accumulator.shape)

    top_k = 10
    min_sep_theta = 2 
    min_sep_rho = 10 

    selected_lines = []
    for rho_idx, theta_idx in zip(rho_idx_all, theta_idx_all):
        if accumulator[rho_idx, theta_idx] <= 0:
            break
        theta_val = theta_deg[theta_idx]
        rho_val = nd_rho[rho_idx]                             

        is_distinct = all(
            abs(theta_val - t) > min_sep_theta or abs(rho_val - r) > min_sep_rho
            for t, r in selected_lines
        )
        if is_distinct:
            selected_lines.append((theta_val, rho_val))
        if len(selected_lines) == top_k:
            break

    mean_theta = np.mean([t for t, _ in selected_lines])
    correction_angle = mean_theta - 90.0
    return selected_lines, correction_angle


def main() -> None:
    """Main function."""
    image = cv2.imread("data/parallel.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray, 100, 200)

    height, width = canny_edges.shape
    ys, xs = np.nonzero(canny_edges)
    centers = np.column_stack((xs, ys))

    theta_deg = np.arange(THETA_MIN, THETA_MAX)
    theta_rad = np.deg2rad(theta_deg)

    rho_max = int(np.ceil(np.hypot(width - 1, height - 1)))
    rho_min = -rho_max
    nd_rho = np.arange(rho_min, rho_max)

    accumulator = generate_accumulator(centers, theta_deg, nd_rho)
    top_lines, correction_angle = correct_roll(accumulator, theta_deg, nd_rho)

    print(f"Top-{len(top_lines)} lines (theta, rho):")
    for theta_val, rho_val in top_lines:
        print(f"  theta={theta_val:6.2f} deg  rho={rho_val:7.1f}")
    print(f"Correction angle: {correction_angle:.2f} deg")

    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    corrected = cv2.warpAffine(image, rotation_matrix, (width, height))

    vis = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    diag = max(height, width) * 2
    for theta_val, rho_val in top_lines:
        theta = np.deg2rad(theta_val)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = cos_t * rho_val, sin_t * rho_val
        dx, dy = -sin_t, cos_t
        pt1 = (int(x0 + diag * dx), int(y0 + diag * dy))
        pt2 = (int(x0 - diag * dx), int(y0 - diag * dy))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(canny_edges, cmap='gray'); axes[0].set_title("Canny edges"); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"Top-{len(top_lines)} lines"); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)); axes[2].set_title(f"Corrected ({correction_angle:.2f} deg)"); axes[2].axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(accumulator, cmap='hot', aspect='auto',
                 extent=[theta_deg[0], theta_deg[-1], nd_rho[-1], nd_rho[0]])
    plt.colorbar(label="votes")
    plt.scatter([t for t, _ in top_lines], [r for _, r in top_lines],
                  c='cyan', marker='x', s=60, label="top-k lines")
    plt.xlabel("theta (degrees)"); plt.ylabel("rho (pixels)")
    plt.title("Hough accumulator (roll detection)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
