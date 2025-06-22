from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

k=4

# Load data
if rank == 0:
    df = pd.read_csv(r"C:\Users\hp\Downloads\hierarical_clustering_data_4.csv", header =0)
    data = df.values
else:
    data = None

# Broadcast the data
data = comm.bcast(data, root=0)

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
                    dist = euclidean(data[idx1], data[idx2])
                    if dist < min_dist:
                        min_dist = dist
            pairs.append(((i, j), min_dist))
    return pairs

def parallel_min_distance(pairs):
    """Find minimum pair using MPI reduction"""
    chunk_size = len(pairs)//size
    # Split pairs among processes
    for rank in range(0,size):
        start = rank
        if rank == size-1:
            end = len(pairs)
        else:
            end = (rank+1)*chunk_size

        chunk = pairs[start:end]
    # chunks = np.array_split(pairs,size)
    # chunk = comm.scatter(chunks, root=0)
    local_min = min(chunk, key=lambda x: x[1]) if chunk else ((-1, -1), float('inf'))
    # print(local_min)
    # Find global minimum
    global_min = comm.allreduce(local_min, op=MPI.MIN)

    return global_min

# Main clustering loop
while len(clusters) > k:
    pairs = pairwise_distances(clusters, data)
    (i, j), min_dist = parallel_min_distance(pairs)

    if rank == 0:
        # Merge clusters
        new_cluster = clusters[i].union(clusters[j])
        clusters.pop(j)
        clusters.pop(i)
        clusters.insert(0, new_cluster)

    clusters = comm.bcast(clusters, root=0)

# Final result
if rank == 0:
    # print(f"{clusters}")

    for cluster in range(len(clusters)):
        cluster_1 = list(clusters[0])
        cluster_1_points = data[cluster_1,:]
        if rank ==0 :
            print(cluster_1_points)