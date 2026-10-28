import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# ==============================================================================
# 0. Setup and Image Loading
# ==============================================================================
print("--- 0. Setup: Loading Images ---")

'''
TODO: Load the original image 'bonn.jpg' and noisy image 'bonn_noisy.jpg'
Convert both to grayscale and prepare the noisy image in float format (0-1 range)
Calculate and print the PSNR of the noisy image compared to the original

Note: PSNR with data_range=255 expects 8-bit images (0-255). If you work with
float images in [0, 1], either convert back to uint8 before computing the PSNR
or use data_range=1.0.
'''

# Load images here
original_img_color = cv2.imread('bonn.jpg')  # Load bonn.jpg
original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_RGB2GRAY)   # Convert to grayscale
noisy_img = cv2.cvtColor(cv2.imread("bonn_noisy.jpg"), cv2.COLOR_RGB2GRAY)           # Load bonn_noisy.jpg and convert to grayscale
noisy_img_float_01 = noisy_img / 255.0 # Convert noisy image to float format (0-1)

# Calculate PSNR of noisy image
psnr_noisy = psnr(original_img_gray, noisy_img, data_range=255)

# Display original and noisy images
# TODO: Create a figure showing original and noisy images side by side
cv2.imshow("Original and Noisy grayscale images", cv2.hconcat([original_img_gray, noisy_img]))
cv2.waitKey(0)

# ==============================================================================
# Custom Filter Definitions (for Part 1.1)
# ==============================================================================

def custom_gaussian_filter(image, kernel_size, sigma):
    """
    Custom Gaussian Filter - Implement convolution from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the Gaussian kernel (odd integer)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO: 
    1. Create Gaussian kernel using the formula: G(x,y) = exp(-(x^2 + y^2)/(2*sigma^2))
    2. Normalize the kernel so it sums to 1
    3. Pad the image using reflect mode
    4. Apply convolution manually using nested loops
    """
    half_kernel_size = kernel_size // 2
    ax = np.arange(-half_kernel_size, half_kernel_size + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(image, half_kernel_size, mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.sum(region * kernel)
    return out

def custom_median_filter(image, kernel_size):
    """
    Custom Median Filter - Implement median calculation from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the median filter window (odd integer)
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image using reflect mode
    2. For each pixel, extract the neighborhood window
    3. Calculate the median of the window
    4. Assign the median value to the output pixel
    """
    half_kernel_size = kernel_size // 2
    padded = np.pad(image, half_kernel_size, mode='reflect')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.median(region)
    return out


def custom_bilateral_filter(image, d, sigma_color, sigma_space):
    """
    Custom Bilateral Filter
    
    Args:
        image: Input image (float, 0-1 range)
        d: Diameter of the pixel neighborhood
        sigma_color: Filter sigma in the color space (0-1 range for float images)
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image
    2. For each pixel:
       a. Calculate spatial weights based on distance from center
       b. Calculate range weights based on intensity difference
       c. Combine weights and compute weighted average
    3. Normalize by sum of weights
    """
    half_kernel_size = d // 2
    padded = np.pad(image, half_kernel_size, mode='reflect')
    out = np.zeros_like(image)
    ax = np.arange(-half_kernel_size, half_kernel_size + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+d, j:j+d]
            center = image[i, j]
            range_weight = np.exp(-((region - center)**2) / (2 * sigma_color**2))
            weights = spatial_weight * range_weight
            out[i, j] = np.sum(region * weights) / np.sum(weights)
    return out

# ==============================================================================
# 1. Filter Application (Part 1.1)
# ==============================================================================
print("\n--- 1.1 Filter Application ---")

# Default Parameters
K_DEFAULT = 7
S_DEFAULT = 2.0
D_DEFAULT = 9
SC_DEFAULT = 100  # cv2 range (0-255)
SS_DEFAULT = 75

# -------------------------- Gaussian Filter --------------------------
print("Applying Gaussian Filter...")
'''
TODO: 
1. Apply Gaussian filter using cv2.GaussianBlur()
2. Apply your custom Gaussian filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots (noisy, cv2 result, custom result)
'''

denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img, (K_DEFAULT, K_DEFAULT), S_DEFAULT)
psnr_gaussian_cv2 = psnr(original_img_gray, denoised_gaussian_cv2, data_range=255)

denoised_gaussian_custom = (custom_gaussian_filter(noisy_img_float_01, K_DEFAULT, S_DEFAULT) * 255).clip(0, 255).astype(np.uint8)
psnr_gaussian_custom = psnr(original_img_gray, denoised_gaussian_custom, data_range=255)

print("Gaussian psnr: custom->", psnr_gaussian_custom, " cv->", psnr_gaussian_cv2)
# Display results here
plt.figure(figsize=(12,4))
for idx, (title, img) in enumerate([("Noisy", noisy_img), ("cv2", denoised_gaussian_cv2), ("Custom", denoised_gaussian_custom)]):
    plt.subplot(1,3,idx+1); plt.imshow(img, cmap='gray'); plt.title(title); plt.axis('off')
plt.show()



# -------------------------- Median Filter --------------------------
print("Applying Median Filter...")
'''
TODO:
1. Apply Median filter using cv2.medianBlur()
2. Apply your custom Median filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_median_cv2 = cv2.medianBlur(noisy_img, K_DEFAULT)
psnr_median_cv2 = psnr(original_img_gray, denoised_median_cv2, data_range=255)

denoised_median_custom = (custom_median_filter(noisy_img_float_01, K_DEFAULT) * 255).clip(0, 255).astype(np.uint8)
psnr_median_custom = psnr(original_img_gray, denoised_median_custom, data_range=255)

print("Median psnr: custom->", psnr_median_custom, " cv->", psnr_median_cv2)
# Display results here
plt.figure(figsize=(12,4))
for idx, (title, img) in enumerate([("Noisy", noisy_img), ("cv2", denoised_median_cv2), ("Custom", denoised_median_custom)]):
    plt.subplot(1,3,idx+1); plt.imshow(img, cmap='gray'); plt.title(title); plt.axis('off')
plt.show()

# -------------------------- Bilateral Filter --------------------------
print("Applying Bilateral Filter...")
'''
TODO:
1. Apply Bilateral filter using cv2.bilateralFilter()
2. Apply your custom Bilateral filter (remember to scale sigma_color for 0-1 range)
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_bilateral_cv2 = cv2.bilateralFilter(noisy_img, D_DEFAULT, SC_DEFAULT, SS_DEFAULT)
psnr_bilateral_cv2 = psnr(original_img_gray, denoised_bilateral_cv2, data_range=255)

denoised_bilateral_custom = (custom_bilateral_filter(noisy_img_float_01, D_DEFAULT, SC_DEFAULT / 255.0, SS_DEFAULT) * 255).clip(0, 255).astype(np.uint8)
psnr_bilateral_custom = psnr(original_img_gray, denoised_bilateral_custom, data_range=255)

print("Bilateral psnr: custom->", psnr_bilateral_custom, " cv->", psnr_bilateral_cv2)
# Display results here
plt.figure(figsize=(12,4))
for idx, (title, img) in enumerate([("Noisy", noisy_img), ("cv2", denoised_bilateral_cv2), ("Custom", denoised_bilateral_custom)]):
    plt.subplot(1,3,idx+1); plt.imshow(img, cmap='gray'); plt.title(title); plt.axis('off')
plt.show()


# ==============================================================================
# 2. Performance Comparison (Part 1.2)
# ==============================================================================
print("\n--- 1.2 Performance Comparison ---")
'''
TODO:
1. Compare PSNR values of all three filters
2. Determine which filter performs best
3. Display side-by-side comparison of all filtered images
4. Print the results with the best performing filter highlighted
'''
results = {"Gaussian": psnr_gaussian_custom, "Median": psnr_median_custom, "Bilateral": psnr_bilateral_custom}
best = max(results, key=results.get)
print(results, "-> best:", best)


# ==============================================================================
# 3. Parameter Optimization (Part 1.3)
# ==============================================================================

def run_optimization(original_img, noisy_img):
    """
    Optimize parameters for all three filters to maximize PSNR
    
    Args:
        original_img: Original clean image
        noisy_img: Noisy image to be filtered
    
    Returns:
        Dictionary containing optimal parameters and best PSNR for each filter
    
    TODO:
    1. For Gaussian filter: iterate over kernel_sizes and sigma values
    2. For Median filter: iterate over kernel_sizes
    3. For Bilateral filter: iterate over d, sigma_color, and sigma_space values
    4. Track the best PSNR and corresponding parameters for each filter
    5. Return results as a dictionary
    
        """
    noisy_float = noisy_img / 255.0
    best = {}

    guassian_best_psnr, guassian_best_k, guassian_best_s = -1, None, None
    for k in [3, 5, 7, 9]:
        for s in [0.5, 1.0, 2.0, 3.0]:
            out = (custom_gaussian_filter(noisy_float, k, s) * 255).clip(0,255).astype(np.uint8)
            p = psnr(original_img, out, data_range=255)
            if p > guassian_best_psnr:
                guassian_best_psnr, guassian_best_k, guassian_best_s = p, k, s
    best["gaussian"] = {"kernel_size": guassian_best_k, "sigma": guassian_best_s, "psnr": guassian_best_psnr}

    median_best_psnr, median_best_k = -1, None
    for k in [3, 5, 7, 9]:
        out = (custom_median_filter(noisy_float, k) * 255).clip(0,255).astype(np.uint8)
        p = psnr(original_img, out, data_range=255)
        if p > median_best_psnr:
            median_best_psnr, median_best_k = p, k
    best["median"] = {"kernel_size": median_best_k, "psnr": median_best_psnr}

    bilateral_best_psnr, bilateral_best_d, bilateral_best_sc, bilateral_best_ss = -1, None, None, None
    for d in [5, 7, 9, 11]:
          for sc in [0.05, 0.1, 0.2, 0.3]:
              for ss in [25, 50, 75, 100]:
                out = (custom_bilateral_filter(noisy_float, d, sc, ss) * 255).clip(0,255).astype(np.uint8)
                p = psnr(original_img, out, data_range=255)
                if p > bilateral_best_psnr:
                    bilateral_best_psnr, bilateral_best_d, bilateral_best_sc, bilateral_best_ss = p, d, sc, ss
    best["bilateral"] = {"d": bilateral_best_d, "sigma_color": bilateral_best_sc, "sigma_space": bilateral_best_ss, "psnr": bilateral_best_psnr}
    return best


'''
TODO:
1. Call run_optimization() function
2. Extract optimal parameters for each filter
3. Apply filters using optimal parameters
4. Display the optimized results in a 2x2 grid (noisy + 3 optimal filters)
5. Print the optimal parameters clearly
'''
# best_params = run_optimization(original_img_gray, noisy_img)
best_params = {"gaussian": {'kernel_size': 9, 'sigma': 1.0, 'psnr': np.float64(22.935385590059646)},
  "median": {'kernel_size': 7, 'psnr': np.float64(22.05998939083642)},
  "bilateral": {'d': 5, 'sigma_color': 0.3, 'sigma_space': 25, 'psnr': np.float64(22.85829545391507)}
}

print("\nOptimal Parameters Found:")
for filter_name, params in best_params.items():
    print(f"  {filter_name}: {params}")

gaussian_opt = best_params["gaussian"]
median_opt = best_params["median"]
bilateral_opt = best_params["bilateral"]

denoised_gaussian_opt = (custom_gaussian_filter(
    noisy_img_float_01, gaussian_opt["kernel_size"], gaussian_opt["sigma"]
) * 255).clip(0, 255).astype(np.uint8)
denoised_median_opt = (custom_median_filter(
    noisy_img_float_01, median_opt["kernel_size"]
) * 255).clip(0, 255).astype(np.uint8)
denoised_bilateral_opt = (custom_bilateral_filter(
    noisy_img_float_01, bilateral_opt["d"], bilateral_opt["sigma_color"], bilateral_opt["sigma_space"]
) * 255).clip(0, 255).astype(np.uint8)

plt.figure(figsize=(10, 10))
images = [
    ("Noisy", noisy_img),
    (f"Optimal Gaussian (k={gaussian_opt['kernel_size']}, sigma={gaussian_opt['sigma']})\nPSNR={gaussian_opt['psnr']:.2f}", denoised_gaussian_opt),
    (f"Optimal Median (k={median_opt['kernel_size']})\nPSNR={median_opt['psnr']:.2f}", denoised_median_opt),
    (f"Optimal Bilateral (d={bilateral_opt['d']}, sc={bilateral_opt['sigma_color']}, ss={bilateral_opt['sigma_space']})\nPSNR={bilateral_opt['psnr']:.2f}", denoised_bilateral_opt),
]
for idx, (title, img) in enumerate(images):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()


# ==============================================================================
# 4. Discussion (Part 1.4)
# ==============================================================================
'''
TODO: Write your discussion (3-5 sentences) in the PDF submission.
'''
