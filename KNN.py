# import numpy as np

# dataset_size = 10000
# num_features = 10
# k = 7
# num_classes = 3
# np.random.seed(42)

# synthetic_dataset = np.random.rand(dataset_size, num_features)
# euclidean_distances = np.zeros((dataset_size, 1))
# query_datapoint = np.random.rand(1, num_features)

# synthetic_labels = np.random.randint(0, num_classes, size=(dataset_size, 1))

# synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis=1)

# for i in range(dataset_size) :
#     current_distance = np.linalg.norm(query_datapoint - synthetic_dataset[i,:-1])
#     euclidean_distances[i, :] = current_distance
# # Sort the distances list to get the labels of the minimum k distances
# sorted_indices = np.argsort(euclidean_distances, axis=0).flatten()
# top_k_instances = synthetic_dataset[sorted_indices[:k]]
# top_k_labels = top_k_instances[:, -1]
# print(f"the class of the query instance is {np.bincount(top_k_labels.astype(int)).argmax()}")

########################### Not distributed
import numpy as np

np.random.seed(42)

# dataset_size = 10000
# num_features = 7
# n_neighbors = 5
# num_classes = 3

# synthetic_dataset = np.random.rand(dataset_size, num_features)
# synthetic_labels = np.random.randint(0, num_classes, size = (dataset_size, 1))
# euclidean_distances = np.zeros((dataset_size,1))

# query_datapoint = np.random.rand(1, num_features)

# synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis = 1)

# for i in range(dataset_size):
#     euc_distance = np.linalg.norm(query_datapoint - synthetic_dataset[i,:-1])
#     euclidean_distances[i,:] = euc_distance

# sorted_indices = np.argsort(euclidean_distances, axis=0).flatten()
# top_n_labels = synthetic_dataset[sorted_indices[:n_neighbors]][:,-1]

# # print(top_n_labels)
# print(f"The class of the query datapoint is {np.bincount(top_n_labels.astype(int)).argmax()}")

############################# Collective communication
# import numpy as np
# from mpi4py import MPI
# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# dataset_size = 10000
# num_features = 7
# n_neighbors = 5
# num_classes = 3


# if my_rank == 0:
#     synthetic_dataset = np.random.rand(dataset_size, num_features)
#     synthetic_labels = np.random.randint(0, num_classes, size = (dataset_size, 1))
#     query_datapoint = np.random.rand(1, num_features)
#     synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis = 1)

#     chunk_size = dataset_size // num_ranks
#     dataset_chunks = []
#     for rank in range(num_ranks):
#         start = rank*chunk_size
#         if rank == num_ranks-1:
#             end = dataset_size
#         else:
#             end = (rank+1)*chunk_size
#         dataset_chunks.append(synthetic_dataset[start:end, :])

# else:
#     synthetic_dataset = None
#     synthetic_labels = None
#     query_datapoint = None
#     chunk_size = None
#     dataset_chunks = None

# query_datapoint = comm.bcast(query_datapoint, root=0)
# dataset_chunk = comm.scatter(dataset_chunks, root=0)

# euclidean_distances = np.zeros((dataset_chunk.shape[0], 2))

# for i in range(dataset_chunk.shape[0]):
#     euc_distances = np.linalg.norm(query_datapoint - dataset_chunk[i,:-1])
#     euclidean_distances[i] = [euc_distances, dataset_chunk[i,-1]]
# sorted_indices = np.argsort(euclidean_distances[:,0], axis=0).flatten()
# sorted_dist = euclidean_distances[sorted_indices[:n_neighbors]]

# final_dist_labels = comm.gather(sorted_dist, root=0)

# if my_rank == 0:
#     all_distances = np.vstack(final_dist_labels)
#     global_sorted_indices = np.argsort(all_distances[:,0], axis=0).flatten()
#     top_k_global_labels = all_distances[global_sorted_indices[:n_neighbors]][:,-1]

#     print(f"The class of the query point is {np.bincount(top_k_global_labels.astype(int)).argmax()}")

####### Using scatterV
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

my_rank = comm.Get_rank()
num_ranks = comm.Get_size()

dataset_size = 10000
num_features = 7
n_neighbors = 5
num_classes = 3

chunk_size = dataset_size // num_ranks
remainder = dataset_size%num_ranks
counts = [chunk_size+remainder if i==(num_ranks-1) else chunk_size for i in range(num_ranks)]
displacement = [sum(counts[:i]) for i in range(num_ranks)]

if my_rank == 0:
    synthetic_dataset = np.random.rand(dataset_size, num_features)
    synthetic_labels = np.random.randint(0, num_classes, size = (dataset_size, 1))
    query_datapoint = np.random.rand(1, num_features)
    synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis = 1)

else:
    synthetic_dataset = None
    query_datapoint = None

local_data_chunks = np.zeros((counts[my_rank], num_features+1), dtype=np.float64)

if my_rank == 0:
    flat_data = synthetic_dataset

else:
    flat_data = None

query_datapoint = comm.bcast(query_datapoint, root=0)
comm.Scatterv([flat_data,
               tuple(c*(num_features+1) for c in counts),
               tuple(d*(num_features+1) for d in displacement),
               MPI.DOUBLE],
               local_data_chunks,
               root=0)

euclidean_distances = np.zeros((local_data_chunks.shape[0],2))

for i in range(local_data_chunks.shape[0]):
    euc_dist = np.linalg.norm(query_datapoint - local_data_chunks[i,:-1])
    euclidean_distances[i,0] = euc_dist
    euclidean_distances[i,1] = local_data_chunks[i,-1]

if my_rank == 0:
    all_distances = np.zeros((sum(counts),2), dtype=np.float64)

else:
    all_distances = None

comm.Gatherv(euclidean_distances,
            [all_distances,
            tuple(c*2 for c in counts),
            tuple(d*2 for d in displacement),
            MPI.DOUBLE],
            root=0)

if my_rank == 0: 
    sorted_indices = np.argsort(all_distances[:,0],axis=0).flatten()
    final_sorting = all_distances[sorted_indices[:n_neighbors]]
    top_k_labels = final_sorting[:,1]

    print(f"The class of the query point is {np.bincount(top_k_labels.astype(int)).argmax()}")




















# euclidean_distances = np.zeros((dataset_chunk.shape[0], 2))

# for i in range(dataset_chunk.shape[0]):
#     euc_distances = np.linalg.norm(query_datapoint - dataset_chunk[i,:-1])
#     euclidean_distances[i] = [euc_distances, dataset_chunk[i,-1]]
# sorted_indices = np.argsort(euclidean_distances[:,0], axis=0).flatten()
# sorted_dist = euclidean_distances[sorted_indices[:n_neighbors]]

# final_dist_labels = comm.gather(sorted_dist, root=0)

# if my_rank == 0:
#     all_distances = np.vstack(final_dist_labels)
#     global_sorted_indices = np.argsort(all_distances[:,0], axis=0).flatten()
#     top_k_global_labels = all_distances[global_sorted_indices[:n_neighbors]][:,-1]

#     print(f"The class of the query point is {np.bincount(top_k_global_labels.astype(int)).argmax()}")















# for i in range(dataset_size):
#     euc_distance = np.linalg.norm(query_datapoint - synthetic_dataset[i,:-1])
#     euclidean_distances[i,:] = euc_distance

# sorted_indices = np.argsort(euclidean_distances, axis=0).flatten()
# top_n_labels = synthetic_dataset[sorted_indices[:n_neighbors]][:,-1]

# # print(top_n_labels)
# print(f"The class of the query datapoint is {np.bincount(top_n_labels.astype(int)).argmax()}")


