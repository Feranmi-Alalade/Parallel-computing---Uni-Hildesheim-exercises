import numpy as np
import pandas as pd
# from scipy.spatial.distance import euclidean

k=4 # Num of clusters

# Load data

df = pd.read_csv(r"C:\Users\hp\Downloads\hierarical_clustering_data_5.csv", header =0)
data = df.values

# Each point starts in its own cluster
clusters = [{i} for i in range(len(data))]

def pairwise_distances(clusters, data):
    # Compute distances between all pairs of clusters
    pairs = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            # single linkage
            min_dist = float('inf')
            for idx1 in clusters[i]:
                for idx2 in clusters[j]:
                    dist = np.linalg.norm((data[idx1], data[idx2]))
                    if dist < min_dist:
                        min_dist = dist
            pairs.append(((i, j), min_dist))
    return pairs

def min_distance(pairs):
    global_min = min(pairs, key=lambda x: x[1])

    return global_min

# Main clustering loop
while len(clusters) > k:
    pairs = pairwise_distances(clusters, data)
    (i, j), min_dist = min_distance(pairs)

    new_cluster = clusters[i].union(clusters[j])
    clusters.pop(j)
    clusters.pop(i)
    clusters.insert(0, new_cluster)

# Final result
print(f"{clusters}")