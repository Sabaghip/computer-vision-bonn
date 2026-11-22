"""Exercise 2: k-means for texture segmentation.

Implement the functions marked with TODO. Only NumPy, SciPy and Matplotlib may be used;
do NOT use library implementations of k-means (e.g. sklearn.cluster, scipy.cluster.vq).
Run from the code folder:  python ex2_texture_kmeans.py
All figures are saved to results/ex2/.
"""
import itertools
import os
import time

import matplotlib
matplotlib.use("Agg")                       # save figures only, no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# Fixed settings -- do not change
# ----------------------------------------------------------------------------
SEED = 0
N_SEEDS = 10                                # random initializations in (d)
N_TEXTONS = 32
N_TEXTON_SAMPLES = 20000
WINDOWS = [15, 31, 61, 91]
LAMBDAS = [0.0, 0.25, 0.5, 1.0]
MATERIALS = ["brick (vertical)", "brick (horizontal)", "grass", "gravel"]
DATA_DIR = "data"
RESULTS_DIR = os.path.join("results", "ex2")
# ----------------------------------------------------------------------------


# ============================================================================
# (a) k-means
# ============================================================================
def assign_to_centers(X, centers):
    """Assignment step: index of the nearest center for every point.

    X: (N, d) points, centers: (k, d).
    Returns labels (N,) with values in 0..k-1 (ties go to the lower index).
    Use array operations instead of loops over the points.
    Hint: ||x - c||^2 = ||x||^2 - 2 x^T c + ||c||^2.
    """
    # TODO: implement
    raise NotImplementedError


def kmeans(X, k, seed, max_iter=300):
    """k-means as defined in Exercise 1.

    1. Choose k distinct data points as initial centers, using np.random.default_rng(seed).
    2. Repeat: assignment step, then update step (an empty cluster keeps its center).
    3. Stop when an update leaves all centers unchanged (or after max_iter iterations).

    Returns
        labels:  (N,) cluster index of every point
        centers: (k, d) final centers
        J:       final value of the objective J (sum of squared distances to the centers)
    """
    # TODO: implement
    raise NotImplementedError


def kmeans_best_of(X, k, n_init=10, seed=0):
    """Provided: run kmeans with seeds seed, ..., seed+n_init-1 and keep the run with lowest J."""
    runs = [kmeans(X, k, seed + s) for s in range(n_init)]
    return min(runs, key=lambda run: run[2])


# ============================================================================
# (b) Textons
# ============================================================================
def filter_bank(img):
    """Provided: Gaussian-derivative filter bank.

    Scales sigma in {1, 2, 4, 8}; for each scale the smoothed image and the derivatives
    d/dx, d/dy, d2/dx2, d2/dy2, d2/dxdy (scale-normalized). Each channel is standardized.
    Returns responses of shape (H, W, 24).
    """
    responses = []
    for sigma in [1, 2, 4, 8]:
        for order in [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1)]:
            responses.append(ndi.gaussian_filter(img, sigma, order=order) * sigma ** sum(order))
    responses = np.stack(responses, axis=-1)
    return (responses - responses.mean(axis=(0, 1))) / responses.std(axis=(0, 1))


def compute_textons(responses, n_textons, n_samples, seed):
    """Textons = k-means centers of filter-response vectors.

    responses: (H, W, F). Draw n_samples random pixels with np.random.default_rng(seed)
    (without replacement), cluster their response vectors with kmeans_best_of
    (n_init=3, seed=seed) and return the centers, shape (n_textons, F).
    """
    # TODO: implement
    raise NotImplementedError


# ============================================================================
# (c) Texton histograms
# ============================================================================
def texton_histograms(texton_map, n_textons, window):
    """Local texton histograms.

    texton_map: (H, W) texton index of every pixel. For every pixel, return the relative
    frequency of each texton in the window x window neighbourhood centred on that pixel.
    Returns an array of shape (H, W, n_textons) whose entries sum to 1 over the last axis.
    Hint: one-hot encode the texton map and use scipy.ndimage.uniform_filter.
    """
    # TODO: implement
    raise NotImplementedError


# ============================================================================
# Provided helpers
# ============================================================================
def load_data():
    img = np.load(os.path.join(DATA_DIR, "mosaic.npy")).astype(float) / 255.0
    labels = np.load(os.path.join(DATA_DIR, "mosaic_labels.npy")).astype(int)
    return img, labels


def add_position(features, lam):
    """Append the pixel coordinates (x, y), scaled so that lam = 1 gives them the same
    total variance as the features. features: (H, W, d) -> (H, W, d + 2)."""
    H, W, _ = features.shape
    yy, xx = np.mgrid[:H, :W]
    pos = np.stack([xx, yy], axis=-1).astype(float)
    feat_std = np.sqrt(features.reshape(-1, features.shape[-1]).var(axis=0).sum())
    pos_std = np.sqrt(pos.reshape(-1, 2).var(axis=0).sum())
    return np.concatenate([features, lam * feat_std / pos_std * pos], axis=-1)


def segment(features, k, n_init=10, seed=0, stride=4):
    """Cluster the per-pixel features into k segments. For speed, k-means is fitted on every
    stride-th pixel; then every pixel is assigned to the nearest center."""
    H, W, d = features.shape
    X = features.reshape(-1, d)
    _, centers, J = kmeans_best_of(X[::stride], k, n_init=n_init, seed=seed)
    return assign_to_centers(X, centers).reshape(H, W), J


def mean_iou(pred, gt, k=4):
    """Mean IoU over the k classes, using the best matching between cluster indices and
    classes (k-means does not know the class names)."""
    best = 0.0
    for perm in itertools.permutations(range(k)):
        mapped = np.asarray(perm)[pred]
        ious = [np.sum((mapped == c) & (gt == c)) / np.sum((mapped == c) | (gt == c))
                for c in range(k)]
        best = max(best, float(np.mean(ious)))
    return best


MATERIAL_CMAP = ListedColormap(["#c0504d", "#f79646", "#9bbb59", "#4f81bd"])


def show_labels(ax, labels, title, gt=None):
    ax.imshow(labels, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    if gt is not None:
        ax.contour(gt, levels=[0.5, 1.5, 2.5], colors="k", linewidths=0.5)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ============================================================================
# Experiments
# ============================================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()
    img, gt = load_data()
    H, W = img.shape
    log = []

    def report(line):
        print(line)
        log.append(line)

    # --- data overview ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Texture mosaic")
    axes[1].imshow(gt, cmap=MATERIAL_CMAP, vmin=0, vmax=3, interpolation="nearest")
    axes[1].set_title("Ground truth")
    for c, name in enumerate(MATERIALS):
        axes[1].plot([], [], "s", color=MATERIAL_CMAP(c), label=name)
    axes[1].legend(loc="lower left", fontsize=8, framealpha=0.9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "data.png"), dpi=120); plt.close(fig)

    # --- (b) textons ------------------------------------------------------------
    responses = filter_bank(img)
    textons = compute_textons(responses, N_TEXTONS, N_TEXTON_SAMPLES, SEED)
    texton_map = assign_to_centers(responses.reshape(-1, responses.shape[-1]), textons).reshape(H, W)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(responses[..., 6 + 1], cmap="gray"); axes[0].set_title("response d/dx, sigma=2")
    axes[1].imshow(responses[..., 6 + 2], cmap="gray"); axes[1].set_title("response d/dy, sigma=2")
    axes[2].imshow(texton_map, cmap="nipy_spectral", interpolation="nearest")
    axes[2].set_title(f"texton map ({N_TEXTONS} textons)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "b_textons.png"), dpi=120); plt.close(fig)
    usage = np.array([[np.mean(texton_map[gt == c] == t) for t in range(N_TEXTONS)] for c in range(4)])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for c, name in enumerate(MATERIALS):
        ax.plot(range(N_TEXTONS), usage[c], "o-", ms=3, color=MATERIAL_CMAP(c), label=name)
    ax.set_xlabel("texton index"); ax.set_ylabel("relative frequency")
    ax.set_title("texton histogram of each material (whole region)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "b_texton_usage.png"), dpi=120); plt.close(fig)

    # --- (c) segmentation with texton histograms for several window sizes -------
    report("(c) k-means (k=4) on texton histograms")
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(4 * len(WINDOWS), 4.3))
    hists = {}
    for ax, window in zip(axes, WINDOWS):
        hists[window] = texton_histograms(texton_map, N_TEXTONS, window)
        seg, _ = segment(hists[window], 4, seed=SEED)
        score = mean_iou(seg, gt)
        report(f"    window {window:3d}: mIoU = {score:.3f}")
        show_labels(ax, seg, f"window {window}: mIoU {score:.2f}", gt)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "c_windows.png"), dpi=120); plt.close(fig)

    # --- (d) adding the pixel position -------------------------------------------
    window = 61
    report(f"(d) texton histograms (window {window}) + lambda * position")
    fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(4 * len(LAMBDAS), 4.3))
    for ax, lam in zip(axes, LAMBDAS):
        seg, _ = segment(add_position(hists[window], lam), 4, seed=SEED)
        score = mean_iou(seg, gt)
        report(f"    lambda {lam:4.2f}: mIoU = {score:.3f}")
        show_labels(ax, seg, f"lambda = {lam}: mIoU {score:.2f}", gt)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "d_position.png"), dpi=120); plt.close(fig)

    # --- (d) different random initializations (single runs, window 61) ----------
    X_hist = hists[window].reshape(-1, N_TEXTONS)
    report(f"(d) single k-means runs with different seeds (window {window}, no position)")
    single = []
    for s in range(N_SEEDS):
        _, centers, J = kmeans(X_hist[::4], 4, SEED + s)
        seg = assign_to_centers(X_hist, centers).reshape(H, W)
        single.append((J, mean_iou(seg, gt), seg, SEED + s))
        report(f"    seed {SEED + s}: final J = {J:8.2f}, mIoU = {single[-1][1]:.3f}")
    single.sort(key=lambda r: (r[0], r[3]))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
    for ax, (J, score, seg, s) in zip(axes, [single[0], single[-1]]):
        show_labels(ax, seg, f"seed {s}: J = {J:.1f}, mIoU {score:.2f}", gt)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "d_seeds.png"), dpi=120); plt.close(fig)

    report(f"total runtime: {time.time() - t_start:.0f} s")
    with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
