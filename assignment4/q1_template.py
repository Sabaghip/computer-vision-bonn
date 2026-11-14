"""
Task 1: Line Hough transform.
Template for MA-INF 2201 Computer Vision WS26/27
Exercise 03
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
THETA_MIN = 50
THETA_MAX = 270
RHO_MIN = int(-1 * (2 ** 0.5) * 128)
RHO_MAX = int(+1 * (2 ** 0.5) * 128)

def generate_accumulator(edges: np.ndarray, nd_theta: np.ndarray, nd_rho: np.ndarray) -> np.ndarray:
    """Compute accumulator matrix.

    Args:
        edges (np.ndarray): Centers of points.
        nd_theta (np.ndarray): Range of theta values.
        nd_rho (np.ndarray): Range of rho values.

    Returns:
        np.ndarray: Accumulator matrix.
    """
    xs = edges[:, 0].astype(np.float64)
    ys = edges[:, 1].astype(np.float64)
    theta_rad = np.deg2rad(nd_theta)
     
    rho = xs[:, None] * np.cos(theta_rad)[None, :] + ys[:, None] * np.sin(theta_rad)[None, :]
     
    rho_step = nd_rho[1] - nd_rho[0]
    rho_idx = np.round((rho - nd_rho[0]) / rho_step).astype(np.int64)
    rho_idx = np.clip(rho_idx, 0, len(nd_rho) - 1)

    theta_idx = np.broadcast_to(np.arange(len(nd_theta)), rho_idx.shape)

    accumulator = np.zeros((len(nd_rho), len(nd_theta)), dtype=np.int64)
    np.add.at(accumulator, (rho_idx.ravel(), theta_idx.ravel()), 1)
    return accumulator


def main() -> None:
    """Main function."""
    image = cv2.imread("data/control.png", cv2.IMREAD_GRAYSCALE)

    ys, xs = np.nonzero(image > 128)
    centers = np.column_stack((xs, ys))

    nd_theta = np.arange(THETA_MIN, THETA_MAX + 1) 
    nd_rho = np.arange(RHO_MIN, RHO_MAX + 1)        

    accumulator = generate_accumulator(centers, nd_theta, nd_rho)

    min_votes = 3 
    rho_idx, theta_idx = np.nonzero(accumulator >= min_votes)
    detected_theta = nd_theta[theta_idx]
    detected_rho = nd_rho[rho_idx]
    votes = accumulator[rho_idx, theta_idx]

    print(f"Detected {len(detected_theta)} line(s) with >= {min_votes} votes:")
    for theta_deg, rho, v in zip(detected_theta, detected_rho, votes):
        print(f"  theta={theta_deg:>3} deg  rho={rho:>4}  votes={v}")

    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape
    for theta_deg, rho in zip(detected_theta, detected_rho):
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = cos_t * rho, sin_t * rho
        dx, dy = -sin_t, cos_t      
        length = max(h, w) * 2
        pt1 = (int(x0 + length * dx), int(y0 + length * dy))
        pt2 = (int(x0 - length * dx), int(y0 - length * dy))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Detected lines ({len(detected_theta)})")
    axes[0].axis('off')

    im = axes[1].imshow(accumulator, cmap='hot', aspect='auto',
                           extent=[nd_theta[0], nd_theta[-1], nd_rho[-1], nd_rho[0]])
    axes[1].set_xlabel("theta (degrees)")
    axes[1].set_ylabel("rho (pixels)")
    axes[1].set_title("Hough accumulator")
    fig.colorbar(im, ax=axes[1], label="votes")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()