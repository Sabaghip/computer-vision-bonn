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
    # TODO: Compute Accumulator matrix.
    # The implementation here is also will be used in next task. Make implementation generic.
    return None


def main() -> None:
    """Main function."""
    # TODO: Read control image.
    # TODO: Define centers of points and range of theta and rhos.

    accumulator = generate_accumulator(
        centers,
        nd_theta,
        nd_rho,
    )

    # TODO: Perform accumulator thresholding.

    # TODO: Plot lines and hough spaces.



if __name__ == "__main__":
    main()