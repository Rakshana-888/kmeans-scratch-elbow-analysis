# Implementing K-Means from scratch and evaluating with the Elbow Method
# Requirements: numpy, matplotlib, scikit-learn
# Paste and run in a Python environment (Jupyter, .py) with those packages installed.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import time

np.random.seed(42)

#########################
# 1) Generate the data
#########################
n_samples = 500
n_features = 3
true_centers = 4
X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=true_centers, cluster_std=0.70, random_state=42)

print(f"Data shape: {X.shape}; Ground-truth clusters: {np.unique(y_true)}")

#########################
# 2) K-Means implementation (NumPy only for computations)
#########################
def initialize_centers_kmeans_pp(X, K, random_state=None):
    """K-Means++ initialization - returns K initial centers (shape K x dim)"""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    n_samples, dim = X.shape
    centers = np.empty((K, dim), dtype=X.dtype)
    # Choose first center uniformly at random
    idx = rng.randint(0, n_samples)
    centers[0] = X[idx]
    # Choose remaining centers
    closest_dist_sq = np.sum((X - centers[0])**2, axis=1)
    for i in range(1, K):
        # probabilities proportional to distance squared
        probs = closest_dist_sq / closest_dist_sq.sum()
        # choose next center
        r = rng.rand()
        cumulative = np.cumsum(probs)
        next_idx = np.searchsorted(cumulative, r)
        centers[i] = X[next_idx]
        # update closest distances
        dist_sq_to_new = np.sum((X - centers[i])**2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_to_new)
    return centers

def kmeans_numpy(X, K, max_iters=300, tol=1e-4, init='kmeans++', random_state=None):
    """
    K-Means clustering using NumPy. Returns best labels, centers, inertia, and number of iterations.
    Single-run (no n_init here). Supports 'kmeans++' and 'random' initializations.
    """
    n_samples, dim = X.shape
    rng = np.random.RandomState(random_state)
    if init == 'kmeans++':
        centers = initialize_centers_kmeans_pp(X, K, random_state=random_state)
    elif init == 'random':
        indices = rng.choice(n_samples, K, replace=False)
        centers = X[indices].copy()
    else:
        raise ValueError("Unknown init: choose 'kmeans++' or 'random'")
    
    labels = np.zeros(n_samples, dtype=int)
    for it in range(1, max_iters+1):
        # Assignment step: compute squared distances and assign labels
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n_samples, K, dim)
        dist_sq = np.sum(diff**2, axis=2)  # (n_samples, K)
        new_labels = np.argmin(dist_sq, axis=1)
        
        # Update step: recompute centers
        new_centers = np.zeros_like(centers)
        for k in range(K):
            members = X[new_labels == k]
            if len(members) == 0:
                # Empty cluster: reinitialize this center to a random data point
                new_centers[k] = X[rng.randint(0, n_samples)]
            else:
                new_centers[k] = members.mean(axis=0)
        
        # Convergence check: centers movement
        center_shift = np.sqrt(np.sum((new_centers - centers)**2, axis=1))
        centers = new_centers
        labels = new_labels
        
        if np.all(center_shift <= tol):
            return labels, centers, np.sum(np.min(dist_sq, axis=1)), it
    # If did not converge within max_iters
    final_inertia = np.sum(np.min(dist_sq, axis=1))
    return labels, centers, final_inertia, max_iters

def kmeans_with_restarts(X, K, n_init=10, init='kmeans++', random_state=None):
    """Run kmeans multiple times and keep the best result by inertia"""
    best_inertia = np.inf
    best_result = None
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 1000000, size=n_init)
    for i, seed in enumerate(seeds):
        labels, centers, inertia, nit = kmeans_numpy(X, K, init=init, random_state=int(seed))
        if inertia < best_inertia:
            best_inertia = inertia
            best_result = (labels.copy(), centers.copy(), inertia, nit)
    return best_result

#########################
# 3) Run Elbow Method K=1..10
#########################
Ks = list(range(1, 11))
inertias = []
results_by_k = {}

start_time = time.time()
for K in Ks:
    labels, centers, inertia, nit = kmeans_with_restarts(X, K, n_init=10, init='kmeans++', random_state=42)
    inertias.append(inertia)
    results_by_k[K] = {'labels': labels, 'centers': centers, 'inertia': inertia, 'n_iter': nit}
    print(f"K={K:2d}  inertia={inertia:.4f}  iterations={nit}")
end_time = time.time()
print(f"Elapsed time (s): {end_time - start_time:.2f}")

#########################
# 4) Choose best K (visual inspection of elbow). We'll print inertias, and select K where
#    inertia drop slows down. Here we also compute the "percentage drop" to help decision.
#########################
inertias = np.array(inertias)
drops = -np.diff(inertias)  # positive numbers represent decrease in inertia when increasing K
percent_drops = drops / inertias[:-1] * 100.0
print("\nInertia by K:")
for K, inertia in zip(Ks, inertias):
    print(f" K={K:2d} -> inertia={inertia:.4f}")
print("\nPercent drops when increasing K by 1:")
for K, pd in zip(Ks[1:], percent_drops):
    print(f" K={K-1}->{K}: drop={pd:.2f}%")

# Heuristic: choose K with largest second derivative (elbow). Compute "knee" using maximum of (drop_{K-1}-drop_{K}) 
second_diffs = np.diff(drops)  # change in drop magnitude
if len(second_diffs) > 0:
    elbow_index = np.argmax(second_diffs) + 1  # +1 because second_diffs index 0 corresponds to K=2 compared to K=1->2
    chosen_K = Ks[elbow_index]
else:
    chosen_K = 1

print(f"\nHeuristic chosen K = {chosen_K} (using second-derivative heuristic).")

#########################
# 5) Visualize Elbow and clustering result (2 plots)
#########################
# Elbow plot
plt.figure(figsize=(6,4))
plt.plot(Ks, inertias, marker='o')
plt.title("Elbow Plot: Inertia vs K")
plt.xlabel("K (number of clusters)")
plt.ylabel("Inertia (sum of squared distances)")
plt.xticks(Ks)
plt.grid(True)
plt.show()

# 2D scatter via PCA colored by cluster labels for chosen K
chosen_result = results_by_k[chosen_K]
labels_chosen = chosen_result['labels']

pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X)

plt.figure(figsize=(6,5))
plt.scatter(X2[:,0], X2[:,1], c=labels_chosen, s=30)  # let matplotlib pick colors
plt.title(f"Data projected to 2D (PCA) and colored by K={chosen_K} cluster assignments")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.grid(True)
plt.show()

# Print brief summary for submission
print("\nSummary for submission:")
print("-----------------------")
print(f"Best K (heuristic): {chosen_K}")
print("Inertia list (K=1..10):")
for K, inertia in zip(Ks, inertias):
    print(f" K={K}: inertia={inertia:.4f}")
print("\nNote: Ground-truth clusters were 4. The elbow and inertia numbers should help justify choice of K.")
