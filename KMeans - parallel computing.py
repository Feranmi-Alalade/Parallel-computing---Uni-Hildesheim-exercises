import numpy as np
import pandas as pd
from mpi4py import MPI
import matplotlib.pyplot as plt

# np.random.seed(14)
comm = MPI.COMM_WORLD

my_rank = comm.Get_rank() # ID of rank being called
num_ranks = comm.Get_size() # Number of processes
k = 4 # number of clusters
max_iterations = 100 # Maximum number of new-centroid computations
min_iterations = 90
tolerance = 1e-6 # Maximum difference between old and new centroids

############################ 3. KMeans
# Initialize clusters
if my_rank == 0:
    ############################ 1. Load the Data
    dataset = pd.read_csv(r"C:\Users\hp\Downloads\First semester\Distributed Data Analytics\cluster_data.csv", header=0, index_col=0) # Load dataset
    # Calcualte the shape of the dataset
    n_samples = dataset.shape[0]
    n_columns = dataset.shape[1]
    # Plotting the data
    plt.scatter(x=dataset.values[:,0], y=dataset.values[:,1])
    plt.title("Points before clustering")
    plt.show()

    data_array = dataset.values # Turn dataset into array
    # CHoose k random indices from the dataset for selecting random centroids
    centroid_indices = np.random.choice(data_array.shape[0], k, replace=False)
    centroids = data_array[centroid_indices,:]

    # Calculate the size of locals
    chunk_size = n_samples//num_ranks # Calculate chunk size
    remainder = n_samples%num_ranks # Remainder

    counts = [chunk_size+remainder if i == (num_ranks-1) else chunk_size for i in range(num_ranks)] # size of each rank
    displacements = [sum(counts[:i]) for i in range(num_ranks)] # displacement offset of each rank

else:
    data_array = None
    centroids = None
    n_samples = None
    n_columns = None
    counts = None
    displacements = None

# Broadcast all necessary data
n_samples = comm.bcast(n_samples, root=0)
n_columns = comm.bcast(n_columns, root=0)
centroids = comm.bcast(centroids, root=0)
counts = comm.bcast(counts, root=0)
displacements = comm.bcast(displacements, root=0)

#Create the recv buffer
local_data_chunks = np.zeros((counts[my_rank], n_columns), dtype=np.float64) # Create buffer to receive local data chunks

# Scatterv to distribute the data array amoung the processes
comm.Scatterv([data_array,
               tuple(c*n_columns for c in counts), # Converting to 1d array, c times number of columns
               tuple(d*n_columns for d in displacements), # same for displacement
               MPI.DOUBLE],
               local_data_chunks, root=0)


# Initializing the distance matrix for the distance between each datapoint and each cluster
# Distance matrix is of shape N x k
local_centroid_dists = np.zeros((local_data_chunks.shape[0],k))

# max_interations number of Iterations
for iteration in range(max_iterations):
    # Iterating through all points in the local chunk
    for i in range(local_data_chunks.shape[0]):
        # Iterating through the clusters
        for j in range(centroids.shape[0]):
            local_centroid_dists[i,j] = np.linalg.norm(local_data_chunks[i,:]-centroids[j,:])

    # Return the index (labels) of each datapoint which is the index with the least distance to a centroid
    labels = np.argmin(local_centroid_dists, axis = 1).astype(np.int32)

    # Initialize the sum of points in each cluster, shape k X Num columns
    new_local_cluster_sum = np.zeros((k, local_data_chunks.shape[1]), dtype=np.float64)
    counts_per_cluster = np.zeros(k, dtype=np.int32)

    for cluster in range(k):
        cluster_points = local_data_chunks[labels==cluster]
        if len(cluster_points) >0:
            new_local_cluster_sum[cluster] = cluster_points.sum(axis=0)
            counts_per_cluster[cluster] = len(cluster_points)

        else:
            new_local_cluster_sum[cluster,:] = centroids[cluster,:]


    # Do a summation of the points in each cluster
    global_centroids_sum = comm.reduce(new_local_cluster_sum, op=MPI.SUM, root=0)
    global_count = comm.reduce(counts_per_cluster, op=MPI.SUM, root=0)

    # Initialize new centroids, shape k x num of columns
    new_global_centroids = np.zeros((k, local_data_chunks.shape[1]),dtype=np.float64)

    if my_rank == 0:
        for cluster in range(k):
            if global_count[cluster]>0:
                new_global_centroids[cluster] = global_centroids_sum[cluster]/global_count[cluster] # Calculate the mean as the new centroids
            # Empty clusters, assign old centroids as new centroids
            else:
                new_global_centroids[cluster] = centroids[cluster]
        
        converged = np.linalg.norm(centroids-new_global_centroids) < tolerance
        centroids = new_global_centroids
    
    else:
        converged = None
        global_labels = None

    # Broadcast converged status and centroid sum to all ranks
    converged = comm.bcast(converged, root=0)
    centroids = comm.bcast(centroids, root=0)

    if converged and iteration >= min_iterations:
        if my_rank == 0:
            print(f"Converged after {iteration} iterations")
            print(global_count)

            # Initialize global labels to gather local labels
            global_labels = np.zeros(data_array.shape[0], dtype=np.int32)

        else:
            global_labels = None

        # Gather labels
        comm.Gatherv(labels,
            [global_labels, counts,
            displacements, MPI.INT],
            root=0)
        
        if my_rank == 0:
            # print(global_labels.shape)
            # print(data_array.shape)
            # COncatenate the data array and the labels
            clustered_data = np.hstack((data_array, global_labels.reshape(-1,1)))
            # Print global count
            for cluster in range(k):
                print(f"The count of cluster {cluster} is {global_count[cluster]}")

            print(clustered_data.shape)

            # Plot the scatter chart
            plt.scatter(x=clustered_data[:,0], y=clustered_data[:,1], c=clustered_data[:,2])
            plt.title("Cluster Assignments")
            plt.show()
        break