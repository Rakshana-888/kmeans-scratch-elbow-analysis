# kmeans-scratch-elbow-analysis

K-Means Clustering From Scratch (NumPy Only)

This project implements the K-Means clustering algorithm entirely from scratch using only NumPy for numerical operations. A synthetic 3D dataset with four well-separated clusters is generated and used to evaluate the algorithm’s performance, convergence behavior, and optimal value of K using the Elbow Method.

📌 Project Overview

This project demonstrates a full, manual implementation of the K-Means algorithm, including:

Random centroid initialization

Distance computation

Cluster assignment step

Centroid update step

Convergence checking (centroids stop changing)

Inertia (sum of squared distances) calculation

The implementation is then applied to a synthetic dataset generated using sklearn.datasets.make_blobs.

📊 Key Features

Custom K-Means Implementation: No use of scikit-learn’s KMeans class.

Synthetic Dataset: 500 samples, 3 features, 4 true clusters.

Elbow Method: Runs K from 1 to 10 and records inertia.

Cluster Visualization: 2D scatter plot using PCA projection or selected axes.

Elbow Curve Plot: Shows inertia vs. number of clusters.

📁 Project Contents

kmeans.py — Fully custom K-Means implementation

analysis.py — Dataset generation, elbow method, and visualizations

elbow_method_output.txt — Text-based inertia results

kmeans_analysis.txt — Written interpretation of clustering quality

📈 Results Summary

The Elbow Method shows a sharp drop in inertia from K=1 → K=4.

After K=4, the decrease becomes minimal, forming a clear “elbow.”

Optimal number of clusters: K = 4, matching the true dataset structure.

The custom implementation accurately separates the four ground-truth clusters.

📝 Written Analysis

The clustering results show strong separation and compactness within clusters. The K-Means algorithm converged reliably, and inertia values validated the correct number of clusters. A brief written analysis is included in kmeans_analysis.txt.

🚀 How to Run
pip install numpy matplotlib scikit-learn
python analysis.py

📚 Learning Outcomes

Understanding how K-Means works internally

Implementing clustering logic manually

Exploring convergence behavior

Using the Elbow Method for model selection

Visualizing clusters and inertia curves

📄 License

This project is released for educational and academic use.
