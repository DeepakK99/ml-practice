import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# Create synthetic customer-like data
X, _ = make_blobs(
    n_samples=300,
    centers=[
        [25, 2000],
        [40, 5000],
        [60, 9000],
    ],
    cluster_std=5,
    random_state=42,
)


# Train K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto",
)

kmeans.fit(X)


# Cluster assignments
labels = kmeans.labels_

# Cluster centers
centers = kmeans.cluster_centers_


print("Cluster centers:")
print(centers)

print("\nFirst 20 cluster assignments:")
print(labels[:20])


# Visualize
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
)

plt.xlabel("Age")
plt.ylabel("Annual Spending")
plt.title("Customer Segmentation with K-Means")

plt.show()