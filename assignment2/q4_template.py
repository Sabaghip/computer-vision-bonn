import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. Setup: Loading the Low-Contrast Image
# ==============================================================================
print("--- 0. Setup: Loading Image ---")

'''
TODO: Load the image 'bonn_lowcontrast.png' as a grayscale image (uint8, values 0-255)
'''

low_contrast_img = cv2.imread('bonn_lowcontrast.png', cv2.IMREAD_GRAYSCALE)

print(f"Image loaded successfully. Size: {low_contrast_img.shape}")
print(f"Intensity range: [{low_contrast_img.min()}, {low_contrast_img.max()}]")

# ==============================================================================
# 1. Histogram and CDF (Part 4.1)
# ==============================================================================
print("\n--- 4.1 Histogram and CDF ---")


def compute_histogram(img):
    """
    Compute the normalized histogram p_f(y) of a grayscale image.
    Do NOT use cv2.calcHist, np.histogram or np.bincount.

    Args:
        img: Input grayscale image (uint8)

    Returns:
        p_f: Array of length 256, p_f[y] = fraction of pixels with value y

    TODO:
    1. Create an array of 256 zeros
    2. Count how many pixels have each gray value y
    3. Normalize by the total number of pixels so that the values sum to 1
    """
    p_f = np.zeros(256)
    total = img.size
    for y in range(256):
        p_f[y] = np.sum(img == y) / total
    return p_f


def compute_cdf(p_f):
    """
    Compute the cumulative distribution function F_f(y) = sum_{k=0}^{y} p_f(k).

    Args:
        p_f: Normalized histogram (length 256)

    Returns:
        F_f: Array of length 256 with values in [0, 1]

    TODO:
    1. Accumulate the histogram values from gray value 0 up to y
    """
    return np.cumsum(p_f)


p_f = compute_histogram(low_contrast_img)  # Call compute_histogram()
F_f = compute_cdf(p_f)  # Call compute_cdf()

# TODO: Plot the normalized histogram p_f(y) and the CDF F_f(y)
#       (e.g., two subplots, x-axis: gray value 0-255)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(256), p_f); axes[0].set_title("Normalized histogram p_f")
axes[1].plot(F_f); axes[1].set_title("CDF F_f")
plt.show()

# ==============================================================================
# 2. Histogram Equalization from Scratch (Part 4.2)
# ==============================================================================
print("\n--- 4.2 Histogram Equalization ---")


def equalize_histogram(img, F_f):
    """
    Histogram equalization using the mapping h(y) = round(255 * F_f(y)).

    Args:
        img: Input grayscale image (uint8)
        F_f: CDF of the input image (length 256)

    Returns:
        Equalized image (uint8)

    TODO:
    1. Build a lookup table (LUT) of length 256 with h(y) = round(255 * F_f(y))
    2. Convert the LUT to uint8
    3. Apply the LUT to every pixel (use array indexing, no loops over pixels)
    """
    lut = np.round(255 * F_f).astype(np.uint8)
    return lut[img]

equalized_img = equalize_histogram(low_contrast_img, F_f)  # Call equalize_histogram()

# TODO: Display the original and the equalized image side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(low_contrast_img, cmap='gray', vmin=0, vmax=255); 
plt.title("Original (low contrast)"); 
plt.axis('off')
plt.subplot(1, 2, 2); 
plt.imshow(equalized_img, cmap='gray', vmin=0, vmax=255); 
plt.title("Equalized"); plt.axis('off')
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. Comparison with OpenCV (Part 4.3)
# ==============================================================================
print("\n--- 4.3 Comparison with OpenCV ---")

'''
TODO:
1. Apply cv2.equalizeHist() to the low-contrast image
2. Print the maximum absolute pixel difference to your result from 4.2
   (cast to int before subtracting to avoid uint8 overflow)
3. If the results differ, explain the reason in a comment / in your PDF
4. Implement an adapted version that reproduces the OpenCV result
   and print the maximum absolute difference again
'''

equalized_cv2 = cv2.equalizeHist(low_contrast_img)
max_diff = np.max(np.abs(equalized_img.astype(int) - equalized_cv2.astype(int)))

print(f"Max. absolute difference (own vs. cv2): {max_diff}")


def equalize_histogram_opencv_like(img):
    """
    Adapted histogram equalization that reproduces cv2.equalizeHist.

    Args:
        img: Input grayscale image (uint8)

    Returns:
        Equalized image (uint8)

    TODO:
    Modify the mapping from 4.2 so that the result is identical to cv2.equalizeHist
    """
    p_f = compute_histogram(img)
    counts = p_f * img.size
    cdf = np.cumsum(counts)
    cdf_min = cdf[cdf > 0].min()
    lut = np.round((cdf - cdf_min) / (img.size - cdf_min) * 255).astype(np.uint8)
    lut = np.clip(lut, 0, 255)
    return lut[img]


equalized_adapted = equalize_histogram_opencv_like(low_contrast_img)
max_diff_adapted = np.max(np.abs(equalized_adapted.astype(int) - equalized_cv2.astype(int)))

print(f"Max. absolute difference (adapted vs. cv2): {max_diff_adapted}")


# ==============================================================================
# 4. Histogram after Equalization (Part 4.4)
# ==============================================================================
print("\n--- 4.4 Histogram after Equalization ---")

'''
TODO:
1. Compute the normalized histogram of your equalized image (reuse compute_histogram())
2. Plot it
3. Answer in your PDF: why is the histogram not flat?
'''

p_h = compute_histogram(equalized_img)
plt.bar(range(256), p_h); plt.title("Histogram after equalization"); plt.show()

# TODO: Plot the histogram of the equalized image

plt.show()
