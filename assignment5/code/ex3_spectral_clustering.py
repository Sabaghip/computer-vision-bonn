"""Exercise 3: Spectral clustering.

Implement the three functions marked with TODO. Only NumPy and your own k-means from
Exercise 2 (imported below) may be used; do NOT use library implementations of spectral
clustering, spectral embedding or k-means.
Run from the code folder:  python ex3_spectral_clustering.py
All figures are saved to results/ex3/.
"""
import os

import matplotlib
matplotlib.use("Agg")                       # save figures only, no window needed
import matplotlib.pyplot as plt
import numpy as np

from ex2_texture_kmeans import kmeans_best_of     # your k-means from Exercise 2

# ----------------------------------------------------------------------------
# Fixed settings -- do not change
# ----------------------------------------------------------------------------
SEED = 0
N_POINTS = 400
RESULTS_DIR = os.path.join("results", "ex3")
# ----------------------------------------------------------------------------


def make_circles(n_samples=N_POINTS, factor=0.4, noise=0.05, seed=SEED):
    """Provided: two concentric circles (outer radius 1, inner radius factor) with Gaussian noise.
    Returns X (n, 2) and the true labels y (0 = outer, 1 = inner)."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    X = np.vstack([np.c_[np.cos(t_out), np.sin(t_out)],
                   factor * np.c_[np.cos(t_in), np.sin(t_in)]])
    y = np.r_[np.zeros(n_out, int), np.ones(n_in, int)]
    return X + noise * rng.standard_normal(X.shape), y


def make_moons(n_samples=N_POINTS, noise=0.06, seed=SEED):
    """Provided: two interleaving half circles with Gaussian noise. Returns X (n, 2) and y."""
    rng = np.random.default_rng(seed)
    n_a = n_samples // 2
    n_b = n_samples - n_a
    t_a = np.linspace(0, np.pi, n_a)
    t_b = np.linspace(0, np.pi, n_b)
    X = np.vstack([np.c_[np.cos(t_a), np.sin(t_a)],
                   np.c_[1 - np.cos(t_b), 0.5 - np.sin(t_b)]])
    y = np.r_[np.zeros(n_a, int), np.ones(n_b, int)]
    return X + noise * rng.standard_normal(X.shape), y


def compute_affinity(X, sigma):
    """Fully connected Gaussian affinity matrix.

    X: (N, d) data points, sigma: kernel width.
    Returns W of shape (N, N) with W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)) and W_ii = 0.
    """
    # TODO: implement
    raise NotImplementedError


def spectral_embedding(W, k, use_first=True):
    """Laplacian eigenmap of the normalized cut problem (D - W) y = lambda D y.

    Solved via the symmetric problem D^{-1/2} (D - W) D^{-1/2} z = lambda z, y = D^{-1/2} z.
    Returns the embedding Y (N, k) or (N, k-1) if use_first is False,
    and all eigenvalues in ascending order.
    Hint: np.linalg.eigh returns the eigenvalues of a symmetric matrix in ascending order.
    """
    # TODO: implement
    raise NotImplementedError


def spectral_clustering(X, k, sigma, use_first=True):
    """Spectral clustering: affinity -> Laplacian eigenmap -> k-means on the rows.

    Use kmeans_best_of(Y, k, n_init=10, seed=SEED) from Exercise 2 for the last step.
    Returns the cluster labels (N,), the embedding Y and the eigenvalues.
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = {"circles": make_circles()[0], "moons": make_moons()[0]}
    k = 2

    # (d) k-means vs. spectral clustering for several sigma
    sigmas = [0.05, 0.1, 0.3, 1.0]
    fig, axes = plt.subplots(2, 1 + len(sigmas), figsize=(17, 6.5))
    for row, (name, X) in enumerate(datasets.items()):
        km, _, _ = kmeans_best_of(X, k, n_init=10, seed=SEED)
        axes[row, 0].scatter(X[:, 0], X[:, 1], c=km, s=8, cmap="coolwarm")
        axes[row, 0].set_title(f"{name}: k-means")
        for col, sigma in enumerate(sigmas, start=1):
            labels, _, eigvals = spectral_clustering(X, k, sigma)
            axes[row, col].scatter(X[:, 0], X[:, 1], c=labels, s=8, cmap="coolwarm")
            axes[row, col].set_title(f"spectral, $\\sigma$={sigma}")
            print(f"{name:8s} sigma={sigma:<5} smallest eigenvalues: {np.round(eigvals[:3], 4)}")
    for ax in axes.ravel():
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectral_vs_kmeans.png"), dpi=120)
    plt.close(fig)

    # (e) with vs. without the eigenvector of the smallest eigenvalue
    X, y_true = make_circles()
    order = np.argsort(y_true, kind="stable")   # outer circle first, then inner circle
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sigma = 0.2
    labels_with, Y_with, eigvals = spectral_clustering(X, k, sigma, use_first=True)
    labels_without, _, _ = spectral_clustering(X, k, sigma, use_first=False)
    y1 = Y_with[:, 0]
    agree = max(np.mean(labels_with == labels_without), np.mean(labels_with != labels_without))
    print(f"\nsigma={sigma}: lambda_1={eigvals[0]:.1e}, lambda_2={eigvals[1]:.1e}, "
          f"first eigenvector min={y1.min():.4f}, max={y1.max():.4f}")
    print(f"agreement of the two clusterings (up to label permutation): {agree:.3f}")

    axes[0].plot(y1[order], ".", ms=3)
    axes[0].set_ylim(y1.mean() - 0.05, y1.mean() + 0.05)
    axes[0].set_title(f"$\\sigma$={sigma}: first eigenvector $y_1$")
    axes[0].set_xlabel("point index (sorted: outer circle, then inner)")
    for ax, lab, title in [(axes[1], labels_with, "with $y_1$ ($y_1, y_2$)"),
                           (axes[2], labels_without, "without $y_1$ ($y_2$ only)")]:
        ax.scatter(X[:, 0], X[:, 1], c=lab, s=8, cmap="coolwarm")
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "first_eigenvector.png"), dpi=120)
    plt.close(fig)
