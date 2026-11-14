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

    # TODO

    # return detected_circles, accumulator

def main():
    """Main function."""
    print("=" * 70)
    print("Task 3: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    # TODO
    
    # Apply Canny edge detection
    # TODO
    
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    # min_radius = 
    # max_radius = 
    # threshold = 
    # min_dist = 
    # r_ssz = 
    # theta_ssz = 
    
    # TODO
    # detected_circles, accumulator = hough_circles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    # TODO
    
    # Visualize accumulator slices
    # TODO
    
    # Visualize peak radius
    # TODO
    
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    # add analysis her.
