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
    # TODO: Sort the accumulater cells by votecount in descending order.

    # TODO: Select the top-k dominant distinct lines and calculate the mean theta of the selected lines.

    # TODO: Calculate correction angle and return the top-k lines and correction angle


def main() -> None:
    """Main function."""
    # cv2 read parallel.jpg and convert to grayscale.
    # TODO: Canny edge
    # canny_edges = cv2.Canny()
    
    height, width = canny_edges.shape
    # TODO: Extract coordinates of the edges (similar to previous task)

    theta_deg = np.arange(THETA_MIN, THETA_MAX)
    theta_rad = np.deg2rad(theta_deg)

    rho_max = int(np.ceil(np.hypot(width - 1, height - 1)))
    rho_min = -rho_max
    nd_rho = np.arange(rho_min, rho_max)

    # TODO: Get the accumulator matrix (use the function from previous task)
    # TODO: Get the lines and correction angle from the correct_roll. 

    # TODO: Compute rotation matrix (cv2.getRotationMatrix2D) and correct image roll (cv2.warpAffine).


    # TODO: Plot the canny edge and top-k lines and unwrapped image.
    # TODO: Plot the hough space image and mark the top-k lines in hough space.

if __name__ == "__main__":
    main()
