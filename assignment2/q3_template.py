import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

'''
TODO: Load the image 'bonn.jpg' and convert it to grayscale
'''

# Load image and convert to grayscale
original_img_color = cv2.imread('bonn.jpg')
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part 3.1)
# ==============================================================================
print("\n--- 3.1 Calculating Integral Image ---")


def calculate_integral_image(img):
    """
    Calculate the integral image (summed area table).
    Each pixel contains the sum of all pixels above and to the left.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Integral image with dimensions (height+1, width+1)
    
    TODO:
    1. Create an integral image array     
    2. Iterate through all pixels and compute integral values
    
        """
    h, w = img.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)

    for i in range(1, h + 1):
        for j in range(1, w + 1):
            integral[i, j] = (
                img[i - 1, j - 1]
                + integral[i - 1, j]
                + integral[i, j - 1]
                - integral[i - 1, j - 1]
            )
    return integral



# Calculate integral image
integral_img = calculate_integral_image(gray_img)  # Call calculate_integral_image()

print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part 3.2)
# ==============================================================================
print("\n--- 3.2 Computing Mean Using Integral Image ---")


def mean_using_integral(integral, top_left, bottom_right):
    """
    Calculate mean gray value using integral image.
    Time Complexity: O(1)

    Args:
        integral: The integral image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract coordinates from top_left and bottom_right
    2. Adjust indices for integral image (remember it's 1-indexed)
    3. Return Sum / number_of_pixels
    """
    y1, x1 = top_left
    y2, x2 = bottom_right
    total = integral[y2+1, x2+1] - integral[y1, x2+1] - integral[y2+1, x1] + integral[y1, x1]
    num_pixels = (y2 - y1 + 1) * (x2 - x1 + 1)
    return total / num_pixels


# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right)  # Call mean_using_integral()

print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")

# ==============================================================================
# 3. Compute Mean by Direct Summation (Part 3.3)
# ==============================================================================
print("\n--- 3.3 Computing Mean by Direct Summation ---")


def mean_by_direct_sum(img, top_left, bottom_right):
    """
    Calculate mean gray value by summing all pixels in region.
    Time Complexity: O(w * h) where w and h are region dimensions

    Args:
        img: The grayscale image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract the region from the image using array slicing
    2. Calculate and return the mean of all pixels in the region
    
      """
    y1, x1 = top_left
    y2, x2 = bottom_right
    region = img[y1:y2+1, x1:x2+1]
    return region.mean()


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)  # Call mean_by_direct_sum()

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part 3.4)
# ==============================================================================
print("\n--- 3.4 Computational Complexity Analysis ---")

'''
TODO:
1. Benchmark both methods by running them multiple times (e.g., 100 iterations)
2. Measure execution time for both methods using time.perf_counter()
3. Compare the execution times
4. Verify that both methods produce the same result
5. Print the results:
   - Method name
   - Average execution time
   - Performance improvement factor


'''

# Benchmark parameters
iterations = 100

print(f"\nBenchmarking with {iterations} iterations...\n")

# TODO: Implement benchmarking code here
start = time.perf_counter()
for _ in range(iterations):
    mean_using_integral(integral_img, top_left, bottom_right)
time_integral = (time.perf_counter() - start) / iterations

start = time.perf_counter()
for _ in range(iterations):
    mean_by_direct_sum(gray_img, top_left, bottom_right)
time_direct = (time.perf_counter() - start) / iterations

# TODO: Display results 
# TODO: Print theoretical complexity explanation
print(f"Integral method: {time_integral*1e6:.2f} µs/call  (O(1))")
print(f"Direct method:   {time_direct*1e6:.2f} µs/call  (O(w*h))")
print(f"Speedup: {time_direct/time_integral:.1f}x")