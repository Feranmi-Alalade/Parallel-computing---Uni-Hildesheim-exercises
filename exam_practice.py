# ######################### Dot product
# 
# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD

# num_ranks = comm.Get_size()
# my_rank = comm.Get_rank()

# def dot_product(a,b):
#     product = 0
#     for i,j in zip(a,b):
#         product += i*j
#     return product

# if my_rank == 0:
#     a = [0,1,2,3,4,5]
#     b = [6,7,8,9,10,11]

#     product = 0
#     chunk_size = len(a)//num_ranks

#     start = 0
#     end = chunk_size
#     a_chunk = a[start:end]
#     b_chunk = b[start:end]

#     product += dot_product(a_chunk,b_chunk)

#     for rank in range(1,num_ranks):
#         start = rank*chunk_size
#         if rank == num_ranks-1:
#             end = len(a)
#         else:
#             end = (rank+1)*chunk_size
        
#         comm.send((a[start:end],b[start:end]), dest=rank)
        
#         dot_prod_local = comm.recv(source=rank)
#         product += dot_prod_local

#     print(product)

# else:
#     a_chunk,b_chunk = comm.recv(source=0)
#     dot_prod_local = dot_product(a_chunk,b_chunk)
#     comm.send(dot_prod_local,dest=0)

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD

# num_ranks = comm.Get_size()
# my_rank = comm.Get_rank()

# def dot_product(a,b):
#     product = 0
#     for i,j in zip(a,b):
#         product += i*j
#     return product

# if my_rank == 0:
#     a = [0,1,2,3,4,5]
#     b = [6,7,8,9,10,11]
#     a_chunks = np.array_split(a, num_ranks)
#     b_chunks = np.array_split(b, num_ranks)

# else:
#     a_chunks = None
#     b_chunks = None

# a_chunk = comm.scatter(a_chunks, root=0)
# b_chunk = comm.scatter(b_chunks, root=0)

# dot_prod = dot_product(a_chunk, b_chunk)

# dot_products = comm.reduce(dot_prod, op=MPI.SUM, root=0)

# if my_rank == 0:
#     print(dot_products)



#################################################### List sorting

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD
# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# def sort_list(list):
#     sorted_list = []
    
#     for i in range(len(list)):
#         min_idx = np.argmin(list)
#         sorted_list.append(list[min_idx])
#         list.pop(min_idx)
#     return sorted_list


# if my_rank == 0:
#     my_list = list(range(10000))
#     np.random.shuffle(my_list)

#     list_chunks = np.array_split(my_list, num_ranks)

# else:
#     list_chunks = None

# list_chunk = comm.scatter(list_chunks, root=0)
# list_chunk = list_chunk.tolist()
# sorted_local = sort_list(list_chunk)

# gathered_sorted_locals = comm.gather(sorted_local, root=0)

# if my_rank == 0:
#     final_list = []
#     for i in range(len(my_list)):
#         first_elements = []
#         for index,local_list in enumerate(gathered_sorted_locals):
#             if local_list:
#                 first_elements.append((local_list[0],index))
#         if first_elements:
#             min_value,min_idx = min(first_elements)

#             final_list.append(min_value)
#             gathered_sorted_locals[min_idx].pop(0)
#     print(final_list)



############################################################################# KNN classification task

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# dataset_size = 10000
# num_features = 10
# num_classes = 3
# k = 3

# if my_rank == 0:
#     synthetic_dataset = np.random.rand(dataset_size, num_features)
#     synthetic_labels = np.random.randint(0, num_classes, size=(dataset_size, 1))
#     synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis=1)
#     query_datapoint = np.random.rand(1, num_features)

#     data_chunks = np.array_split(synthetic_dataset,num_ranks)

# else:
#     synthetic_dataset = None
#     query_datapoint = None
#     data_chunks = None

# query_datapoint = comm.bcast(query_datapoint, root=0)
# data_chunk = comm.scatter(data_chunks, root=0)

# rank_dist_class = np.zeros((data_chunk.shape[0], 2), dtype=np.float64)
# for i in range(data_chunk.shape[0]):
#     dist = np.linalg.norm(query_datapoint - data_chunk[i,:-1])
#     rank_dist_class[i,0] = dist
#     rank_dist_class[i,1] = data_chunk[i,-1]

# print(f"rank: {my_rank}: {rank_dist_class.shape[0]} distances calculated")

# global_dist_class = comm.gather(rank_dist_class, root=0)


# if my_rank == 0:
#     global_dist_class = np.vstack(global_dist_class)
    # sorted_indices = np.argsort(global_dist_class[:,0])
    # top_k_labels = global_dist_class[sorted_indices[:k],1].astype(int)
    # most_common_label = np.bincount(top_k_labels).argmax()
    # print(f"The predicted class of query {query_datapoint} is {most_common_label}")
    # print(top_k_labels)



####################################### KNN regression task

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# dataset_size = 10000
# num_features = 10
# num_classes = 3
# k = 3

# if my_rank == 0:
#     synthetic_dataset = np.random.rand(dataset_size, num_features)
#     synthetic_labels = np.random.rand(dataset_size,1)
#     synthetic_dataset = np.append(synthetic_dataset, synthetic_labels, axis=1)
#     query_datapoint = np.random.rand(1, num_features)

#     data_chunks = np.array_split(synthetic_dataset,num_ranks)

# else:
#     data_chunks = None
#     query_datapoint = None

# query_datapoint = comm.bcast(query_datapoint, root=0)
# data_chunk = comm.scatter(data_chunks, root=0)

# euc_loc_dists = np.zeros((data_chunk.shape[0], 2), dtype=np.float64)
# for i in range(data_chunk.shape[0]):
#     dist = np.linalg.norm(query_datapoint-data_chunk[i,:-1])

#     euc_loc_dists[i,0] = dist
#     euc_loc_dists[i,1] = data_chunk[i,-1]

# euc_global_dists = comm.gather(euc_loc_dists, root=0)

# if my_rank == 0:
#     euc_global_dists = np.vstack(euc_global_dists)

#     sorted_indices = np.argsort(euc_global_dists[:,0])
#     top_k_labels = euc_global_dists[sorted_indices[:k],1]
#     avg_label = np.mean(top_k_labels)

#     print(f"The predicted value of the query point is {avg_label}")




################################################################################################## Distributed KMeans


# from mpi4py import MPI
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# k = 4
# tolerance = 1e-6
# iterations = 1000

# if my_rank == 0:
#     dataset = pd.read_csv(r"C:\Users\hp\Downloads\First semester\Distributed Data Analytics\cluster_data.csv", header=0, index_col=0)
#     dataset = dataset.values
#     centroids_indices = np.random.choice(dataset.shape[0], k, replace=False)
#     centroids = dataset[centroids_indices,:]
#     n_samples = dataset.shape[0]
#     n_columns = dataset.shape[1]

#     data_chunks = np.array_split(dataset,num_ranks)

# else:
#     data_chunks = None
#     centroids = None
#     n_samples = None
#     n_columns = None

# centroids = comm.bcast(centroids,root=0)
# n_samples = comm.bcast(n_samples, root=0)
# n_columns = comm.bcast(n_columns,root=0)

# data_chunk = comm.scatter(data_chunks, root=0)


# centroid_dists = np.zeros((data_chunk.shape[0],k),dtype=np.float64)
# for iteration in range(iterations):
#     for i in range(data_chunk.shape[0]):
#         for j in range(centroids.shape[0]):
#             centroid_dists[i,j] = np.linalg.norm(data_chunk[i,:]-centroids[j,:])

#     labels = np.argmin(centroid_dists,axis=1).astype(int)

#     local_cluster_sum = np.zeros((k,n_columns), dtype=np.float64)
#     local_cluster_count = np.zeros(k,dtype=np.int32)

#     for cluster in range(k):
#         cluster_points = data_chunk[labels==cluster]
#         if len(cluster_points)>0:
#             local_cluster_sum[cluster,:] = cluster_points.sum(axis=0)
#             local_cluster_count[cluster] = len(cluster_points)
#         else:
#             local_cluster_sum[cluster,:] = centroids[cluster,:]

#     global_cluster_sum = comm.reduce(local_cluster_sum,op=MPI.SUM,root=0)
#     global_cluster_count = comm.reduce(local_cluster_count,op=MPI.SUM,root=0)

#     new_centroids = np.zeros((k,n_columns),dtype=np.float64)

#     if my_rank == 0:
#         for cluster in range(k):
#             if global_cluster_count[cluster]>0:
#                 new_centroids[cluster,:] = global_cluster_sum[cluster,:]/global_cluster_count[cluster]
#             else:
#                 new_centroids[cluster,:] = centroids[cluster,:]

#         converged = np.linalg.norm(new_centroids-centroids) < tolerance

#         centroids = new_centroids

#     else:
#         converged = None
#         centroids = None

#     converged = comm.bcast(converged,root=0)
#     centroids = comm.bcast(centroids, root=0)

#     if converged:
#         if my_rank == 0:
#             print(f"Converged after {iteration} iterations")

#         global_labels = comm.gather(labels,root=0)

#         if my_rank == 0:
#             clustered_data = np.hstack((dataset,np.concatenate(global_labels).reshape(-1,1)))
            
#             plt.scatter(x=clustered_data[:,0], y=clustered_data[:,1], c=clustered_data[:,2])

#             plt.show()
#         break





# from mpi4py import MPI
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# k=4
# iterations = 100
# tolerance = 1e-6

# if my_rank == 0:
#     dataset = pd.read_csv(r"C:\Users\hp\Downloads\First semester\Distributed Data Analytics\cluster_data.csv", header=0,index_col=0)
#     dataset = dataset.values

#     data_chunks = np.array_split(dataset,num_ranks)

#     n_cols = dataset.shape[1]
#     n_samples = dataset.shape[0]

#     centroids_indices = np.random.choice(n_samples,k,replace=False)
#     centroids = dataset[centroids_indices,:]

# else:
#     data_chunks = None
#     centroids = None
#     n_cols = None
#     n_samples = None

# centroids = comm.bcast(centroids,root=0)
# n_cols = comm.bcast(n_cols,root=0)
# n_samples = comm.bcast(n_samples,root=0)

# data_chunk = comm.scatter(data_chunks,root=0)
# n_local_samples = data_chunk.shape[0]

# for iteration in range(iterations):
#     loc_centroid_dists = np.zeros((n_local_samples,k),dtype=np.float64)
#     for i in range(n_local_samples):
#         for j in range(k):
#             loc_centroid_dists[i,j] = np.linalg.norm(data_chunk[i,:]-centroids[j,:])
    
#     labels = np.argmin(loc_centroid_dists,axis=1)

#     local_cluster_sum = np.zeros((k,n_cols),dtype=np.float64)
#     local_cluster_count = np.zeros(k,dtype=np.int32)

#     for cluster in range(k):
#         cluster_points = data_chunk[labels==cluster]
#         if len(cluster_points)>0:
#             local_cluster_sum[cluster,:] = np.sum(cluster_points, axis = 0)
#             local_cluster_count[cluster] = len(cluster_points)
#         else:
#             local_cluster_sum[cluster,:] = centroids[cluster,:]

#     global_cluster_sum = comm.reduce(local_cluster_sum, op=MPI.SUM,root=0)
#     global_cluster_count = comm.reduce(local_cluster_count,op=MPI.SUM,root=0)

    

#     if my_rank == 0:
#         global_new_centroids = np.zeros((k,n_cols),dtype=np.float64)
#         for cluster in range(k):
#             if global_cluster_count[cluster]>0:
#                 global_new_centroids[cluster,:] = global_cluster_sum[cluster,:]/global_cluster_count[cluster]
#             else:
#                 global_new_centroids[cluster,:] = centroids[cluster,:]

#         converged = np.linalg.norm(global_new_centroids-centroids)<tolerance

#         centroids = global_new_centroids

#     else:
#         centroids=None
#         converged=None

#     centroids = comm.bcast(centroids,root=0)
#     converged = comm.bcast(converged,root=0)

#     if converged:
#         if my_rank == 0:
#             print(f"Converged after {iteration} iterations")
#             for cluster in range(k):
#                 print(f"CLuster {cluster} has {global_cluster_count[cluster]} points")

#         global_labels = comm.gather(labels,root=0)
            
#         if my_rank == 0:
#             global_labels = np.concatenate(global_labels).reshape(-1,1)

#             clustered_data = np.hstack((dataset,global_labels))
#             plt.scatter(x=clustered_data[:,0],y=clustered_data[:,1],c=clustered_data[:,2])

#             plt.show()
#         break




################################################################################################### Simple Gradient descent

# import matplotlib.pyplot as plt

# list_x = [-4,-3,-2,-1,0,1,2,3,4]

# def y(x):
#     return x**2

# def grad_descent(x):
#     return 2*x

# x = 3.5

# x_list = []
# for i in range(20):
#     x_list.append(x)
#     x -= (0.1*grad_descent(x))

# x_list.append(x)

# print(x_list)

# plt.plot(list_x, [y(x) for x in list_x])
# plt.scatter(x_list, [y(x) for x in x_list], color='red')

# plt.show()



###########################################################################3############### Linear regression using Gradient descent

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# learning_rate = 0.006
# iterations = 10000
# tolerance = 1e-6

# def y_hat(theta1,theta2,X):
#     return (theta1*X)+theta2

# def gradient_descent(theta1,theta2,X,y):
#     # n = len(X)
#     y_pred = y_hat(theta1,theta2,X)

#     theta1_grad = (-2)*np.sum(X*(y-y_pred))
#     theta2_grad = (-2)*np.sum(y-y_pred)

#     return theta1_grad,theta2_grad

# if my_rank == 0:
#     X = np.arange(0,1,0.01)
#     y = X + np.random.normal(0,0.2,len(X))

#     X_chunks = np.array_split(X, num_ranks)
#     y_chunks = np.array_split(y, num_ranks)

#     theta1 = np.random.normal(0,1)
#     theta2 = np.random.normal(0,1)

#     n_samples = len(X)

# else:
#     X_chunks = None
#     y_chunks = None
#     n_samples = None
#     theta1 = None
#     theta2 = None

# n_samples = comm.bcast(n_samples, root=0)
# theta1 = comm.bcast(theta1, root=0)
# theta2 = comm.bcast(theta2,root=0)

# X_chunk = comm.scatter(X_chunks, root=0)
# y_chunk = comm.scatter(y_chunks, root=0)

# for iteration in range(iterations):
#     theta1_grad_local, theta2_grad_local = gradient_descent(theta1,theta2,X_chunk,y_chunk)

#     theta_1_grad_global = comm.reduce(theta1_grad_local, op=MPI.SUM, root=0)
#     theta_2_grad_global = comm.reduce(theta2_grad_local, op=MPI.SUM, root=0)

#     if my_rank == 0:
#         theta_1_grad_global /= n_samples
#         theta_2_grad_global /= n_samples

#         new_theta1 = theta1 - (learning_rate*theta_1_grad_global)
#         new_theta2 = theta2 - (learning_rate*theta_2_grad_global)

#         converged = (np.abs(theta1-new_theta1) < tolerance and np.abs(theta2-new_theta2)<tolerance)
#         theta1 = new_theta1
#         theta2 = new_theta2

#     else:
#         converged = None
#         theta1 = None
#         theta2 = None

#     converged = comm.bcast(converged,root=0)
#     theta1 = comm.bcast(theta1,root=0)
#     theta2 = comm.bcast(theta2,root=0)

#     if converged:
#         if my_rank == 0:
#             print(f"Converged after {iteration} iterations, theta1: {theta1}, theta2: {theta2}")
#         break

# if my_rank == 0 and not converged:
#     print(f"Did not converge after {iterations} iterations. Final theta1: {theta1}, theta2: {theta2}")



############################################################################################### Logistic regression (multi-feature dataset)

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# iterations = 50000
# num_classes = 2
# num_features = 3


# def sigmoid(z):
#     return 1/(1+np.exp(-z))

# def y_hat(theta1,bias,X):
#     return sigmoid(np.dot(X,theta1)+bias)

# def gradient_descent(theta1,bias,X,y):
#     y_pred = y_hat(theta1,bias,X)
#     theta1_grad = np.dot(X.T,(y_pred-y))
#     bias_grad = np.sum(y_pred-y)
#     return theta1_grad, bias_grad

# if my_rank == 0:
#     theta1 = np.random.randn(num_features,1)
#     bias = float(np.random.randn())

#     # X = np.arange(0,1,0.01)
#     X = np.random.rand(10000, num_features)
#     y = np.random.randint(0,num_classes, size = (len(X),1))

#     X_chunks = np.array_split(X,num_ranks)
#     y_chunks = np.array_split(y,num_ranks)
#     learning_rate = 0.006
#     tolerance = 1e-6

# else:
#     theta1 = None
#     bias = None
#     X_chunks = None
#     y_chunks = None
#     learning_rate = None
#     tolerance = None

# theta1 = comm.bcast(theta1, root=0)
# bias = comm.bcast(bias, root=0)
# learning_rate = comm.bcast(learning_rate,root=0)
# tolerance = comm.bcast(tolerance,root=0)
# X_chunk = comm.scatter(X_chunks,root=0)
# y_chunk = comm.scatter(y_chunks,root=0)

# for iteration in range(iterations):
#     theta1_grad_local, bias_grad_local = gradient_descent(theta1,bias,X_chunk,y_chunk)

#     theta1_grad_global = comm.reduce(theta1_grad_local,op=MPI.SUM,root=0)
#     bias_grad_global = comm.reduce(bias_grad_local,op=MPI.SUM,root=0)

#     if my_rank == 0:
#         theta1_grad_global /= len(X)
#         bias_grad_global /= len(X)

#         new_theta1 = theta1 - (learning_rate*theta1_grad_global)
#         new_bias = bias - (learning_rate*bias_grad_global)

#         converged = (np.linalg.norm(new_theta1-theta1)<tolerance and np.linalg.norm(new_bias-bias)<tolerance)
#         theta1 = new_theta1
#         bias = new_bias

#     else:
#         converged = None
#         theta1 = None
#         bias = None
    
#     converged = comm.bcast(converged, root=0)
#     theta1 = comm.bcast(theta1,root=0)
#     bias = comm.bcast(bias, root=0)

#     if converged:
#         if my_rank == 0:
#             print(f"Converged after {iteration} iterations, theta 1 {theta1}, bias {bias}")
#         break

# if not converged and my_rank == 0:
#     print(f"Did not converge after {iterations} iterations, theta 1 {theta1}, bias {bias}")



############################################################################################################## Running hadoop

# docker-compose up --build

# ######### New terminal

# docker ps

# docker exec -it hadoop bash

# root@eeace327577b:/# cd opt
# root@eeace327577b:/opt# cd hadoop/
# root@eeace327577b:/opt/hadoop# cd code
# root@eeace327577b:/opt/hadoop/code# ls
# inpt.txt  mapper.py reducer.py  run.sh

############################################################################################################### SH file to run hadoop

#!/bin/bash

# # Variables
# INPUT_LOCAL="inpt_practice.txt"
# INPUT_HDFS_DIR="/input"
# OUTPUT_HDFS_DIR="/output"
# CONTAINER_NAME="hadoop"  # Replace with your container name
# CONTAINER_WORKDIR="/opt/hadoop/code"

# # Clean previous HDFS files
# hdfs dfs -rm -r -f $INPUT_HDFS_DIR $OUTPUT_HDFS_DIR
# hdfs dfs -mkdir -p $INPUT_HDFS_DIR
# hdfs dfs -put $CONTAINER_WORKDIR/$INPUT_LOCAL $INPUT_HDFS_DIR

# # Run streaming job from inside code directory
# hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar \
#   -input $INPUT_HDFS_DIR \
#   -output $OUTPUT_HDFS_DIR \
#   -mapper "python3 mapper_practice.py" \
#   -reducer "python3 reducer_practice.py" \
#   -file mapper_practice.py \
#   -file reducer_practice.py

# # Step 3: Fetch result back
# hdfs dfs -cat $OUTPUT_HDFS_DIR/part-00000 > $CONTAINER_WORKDIR/output_practice.txt
# echo "MapReduce job complete. Output saved to output_practice.txt"



############################################################################################################ SH file hadoop two mapreduce

#!/bin/bash

# Variables
# INPUT_LOCAL="input_textrank.txt"
# INPUT_HDFS_DIR="/input"
# OUTPUT_HDFS_DIR1="/output1"
# OUTPUT_HDFS_DIR2="/output2"
# CONTAINER_NAME="hadoop"  # Replace with your container name
# CONTAINER_WORKDIR="/opt/hadoop/code"

# # Clean previous HDFS files
# hdfs dfs -rm -r -f $INPUT_HDFS_DIR $OUTPUT_HDFS_DIR1 $OUTPUT_HDFS_DIR2
# hdfs dfs -mkdir -p $INPUT_HDFS_DIR
# hdfs dfs -put $CONTAINER_WORKDIR/$INPUT_LOCAL $INPUT_HDFS_DIR

# # Run streaming job from inside code directory
# hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar \
#   -input $INPUT_HDFS_DIR \
#   -output $OUTPUT_HDFS_DIR1 \
#   -mapper "python3 mapper_textrank.py" \
#   -reducer "python3 reducer_textrank.py" \
#   -file mapper_textrank.py \
#   -file reducer_textrank.py

# hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar \
#   -input $OUTPUT_HDFS_DIR1 \
#   -output $OUTPUT_HDFS_DIR2 \
#   -mapper "python3 mapper_textrank2.py" \
#   -reducer "python3 reducer_textrank2.py" \
#   -file mapper_textrank2.py \
#   -file reducer_textrank2.py

# # Step 3: Fetch result back
# hdfs dfs -cat $OUTPUT_HDFS_DIR2/part-00000 > $CONTAINER_WORKDIR/output_textrank.txt
# echo "MapReduce job complete. Output saved to output_textrank.txt"


################################################################################################################### Naive Bayes

# import numpy as np

# def manual_train_test_split(X, y, train_ratio=0.8):
#     np.random.seed(42)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     split = int(train_ratio * len(X))
#     train_indices = indices[:split]
#     test_indices = indices[split:]
#     return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# def fit_naive_bayes(X_train, y_train):
#     classes = np.unique(y_train)
#     num_features = X_train.shape[1]
#     num_classes = len(classes)

#     priors = np.zeros(num_classes)
#     means = np.zeros((num_classes, num_features))
#     variances = np.zeros((num_classes, num_features))

#     for idx, cls in enumerate(classes):
#         X_cls = X_train[y_train == cls]
#         priors[idx] = len(X_cls) / len(X_train)
#         means[idx] = X_cls.mean(axis=0)
#         variances[idx] = X_cls.var(axis=0) + 1e-9  # avoid division by zero

#     return priors, means, variances, classes

# def gaussian_log_likelihood(x, mean, var):
#     return -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)

# def predict_naive_bayes(X_test, priors, means, variances, classes):
#     y_pred = []
#     for x in X_test:
#         posteriors = []
#         for idx, cls in enumerate(classes):
#             prior_log = np.log(priors[idx])
#             likelihood_log = gaussian_log_likelihood(x, means[idx], variances[idx])
#             posteriors.append(prior_log + likelihood_log)
#         y_pred.append(classes[np.argmax(posteriors)])
#     return np.array(y_pred)

# # --- Main Execution ---

# # Generate dummy data
# np.random.seed(42)
# X = np.random.randn(10000, 3)
# y = np.random.randint(0, 2, size=(10000,))

# # Manual train-test split
# X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

# # Train model
# priors, means, variances, classes = fit_naive_bayes(X_train, y_train)

# # Predict
# y_pred = predict_naive_bayes(X_test, priors, means, variances, classes)

# # Accuracy
# accuracy = np.mean(y_pred == y_test)
# print(f"Test Accuracy: {accuracy:.4f}")


################################## Naive Bayes distribution

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD
# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# def manual_train_test_split(X, y, train_ratio=0.8):
#     np.random.seed(42)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     split = int(train_ratio * len(X))
#     train_idx = indices[:split]
#     test_idx = indices[split:]
#     return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# def compute_mean_std(X):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0) + 1e-9  # avoid division by zero
#     return mean, std

# def standardize(X, mean, std):
#     return (X - mean) / std

# def fit_naive_bayes(X_chunk, y_chunk, num_classes=2):
#     num_features = X_chunk.shape[1]
#     class_counts = np.zeros(num_classes)
#     class_means = np.zeros((num_classes, num_features))
#     class_vars = np.zeros((num_classes, num_features))

#     for cls in range(num_classes):
#         X_cls = X_chunk[y_chunk == cls]
#         class_counts[cls] = len(X_cls)
#         if len(X_cls) > 0:
#             class_means[cls] = np.mean(X_cls, axis=0)
#             class_vars[cls] = np.var(X_cls, axis=0) + 1e-9

#     return class_counts, class_means, class_vars

# def gaussian_log_likelihood(x, mean, var):
#     return -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)

# def predict_naive_bayes(X_test, priors, means, variances):
#     y_pred = []
#     for x in X_test:
#         posteriors = []
#         for idx in range(len(priors)):
#             log_prior = np.log(priors[idx])
#             log_likelihood = gaussian_log_likelihood(x, means[idx], variances[idx])
#             posteriors.append(log_prior + log_likelihood)
#         y_pred.append(np.argmax(posteriors))
#     return np.array(y_pred)

# # --- Main ---
# if my_rank == 0:
#     # Generate separable synthetic data
#     np.random.seed(42)
#     X0 = np.random.normal(0, 1, size=(5000, 3))
#     X1 = np.random.normal(3, 1, size=(5000, 3))
#     X = np.vstack((X0, X1))
#     y = np.array([0]*5000 + [1]*5000)

#     # Standardize
#     mean, std = compute_mean_std(X)
#     X = standardize(X, mean, std)

#     # Split
#     X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

#     # Scatter data
#     X_train_chunks = np.array_split(X_train, num_ranks)
#     y_train_chunks = np.array_split(y_train, num_ranks)

#     X_test_chunks = np.array_split(X_test, num_ranks)
#     y_test_chunks = np.array_split(y_test, num_ranks)

# else:
#     X_train_chunks = None
#     y_train_chunks = None
#     X_test_chunks = None
#     y_test_chunks = None

# # Scatter training data
# X_train_chunk = comm.scatter(X_train_chunks, root=0)
# y_train_chunk = comm.scatter(y_train_chunks, root=0)

# X_test_chunk = comm.scatter(X_test_chunks, root=0)
# y_test_chunk = comm.scatter(y_test_chunks, root=0)

# # Train Naive Bayes
# local_counts, local_means, local_vars = fit_naive_bayes(X_train_chunk, y_train_chunk)

# # Reduce (sum) class counts and calculate global means/variances
# total_counts = np.zeros_like(local_counts)
# total_means = np.zeros_like(local_means)
# total_vars = np.zeros_like(local_vars)

# comm.Reduce(local_counts, total_counts, op=MPI.SUM, root=0)
# comm.Reduce(local_means * local_counts[:, None], total_means, op=MPI.SUM, root=0)
# comm.Reduce(local_vars * local_counts[:, None], total_vars, op=MPI.SUM, root=0)

# if my_rank == 0:
#     priors = total_counts / np.sum(total_counts)
#     means = total_means / total_counts[:, None]
#     variances = total_vars / total_counts[:, None]

# else:
#     priors = None
#     means = None
#     variances = None

# priors = comm.bcast(priors,root=0)
# means = comm.bcast(means,root=0)
# variances = comm.bcast(variances,root=0)

# # Predict on test set
# y_pred = predict_naive_bayes(X_test_chunk, priors, means, variances)
# local_correct = np.sum(y_pred == y_test_chunk)
# local_count = len(y_pred)

# total_correct = comm.reduce(local_correct, op=MPI.SUM, root=0)
# total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# if my_rank == 0:
#     accuracy = total_correct/total_count
#     print(f"Test Accuracy: {accuracy:.4f}")


######################################################################################################## Random Forest distributed

# from mpi4py import MPI
# import numpy as np
# import pickle

# comm = MPI.COMM_WORLD
# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# def train_test_split(X, y, test_size=0.2, random_state=None):
#     if random_state:
#         np.random.seed(random_state)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     split = int(len(X)*(1 - test_size))
#     train_idx, test_idx = indices[:split], indices[split:]
#     return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# def bootstrap_sample(X, y):
#     n_samples = X.shape[0]
#     indices = np.random.choice(n_samples, n_samples, replace=True)
#     return X[indices], y[indices]

# def gini_impurity(y):
#     classes, counts = np.unique(y, return_counts=True)
#     p = counts / counts.sum()
#     return 1 - np.sum(p ** 2)

# def best_split(X, y, feature_subspace):
#     n_features = X.shape[1]
#     features_idx = np.random.choice(n_features, feature_subspace, replace=False)
#     best_feat, best_thresh, best_gain = None, None, -1
#     parent_impurity = gini_impurity(y)
#     n_samples = len(y)

#     for feat in features_idx:
#         thresholds = np.unique(X[:, feat])
#         for thresh in thresholds:
#             left_mask = X[:, feat] <= thresh
#             right_mask = ~left_mask
#             if sum(left_mask) == 0 or sum(right_mask) == 0:
#                 continue
#             left_impurity = gini_impurity(y[left_mask])
#             right_impurity = gini_impurity(y[right_mask])
#             n_left, n_right = sum(left_mask), sum(right_mask)
#             weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
#             gain = parent_impurity - weighted_impurity
#             if gain > best_gain:
#                 best_gain = gain
#                 best_feat = feat
#                 best_thresh = thresh
#     return best_feat, best_thresh, best_gain

# def build_tree(X, y, max_depth, min_samples_split, feature_subspace, depth=0):
#     node = {}

#     num_samples = y.shape[0]
#     num_classes = len(np.unique(y))
#     if (depth >= max_depth) or (num_samples < min_samples_split) or (num_classes == 1):
#         values, counts = np.unique(y, return_counts=True)
#         node['type'] = 'leaf'
#         node['class'] = values[np.argmax(counts)]
#         return node

#     feat, thresh, gain = best_split(X, y, feature_subspace)
#     if gain == -1 or feat is None:
#         values, counts = np.unique(y, return_counts=True)
#         node['type'] = 'leaf'
#         node['class'] = values[np.argmax(counts)]
#         return node

#     left_mask = X[:, feat] <= thresh
#     right_mask = ~left_mask

#     node['type'] = 'node'
#     node['feature'] = feat
#     node['threshold'] = thresh
#     node['left'] = build_tree(X[left_mask], y[left_mask], max_depth, min_samples_split, feature_subspace, depth + 1)
#     node['right'] = build_tree(X[right_mask], y[right_mask], max_depth, min_samples_split, feature_subspace, depth + 1)
#     return node

# def predict_tree(tree, x):
#     if tree['type'] == 'leaf':
#         return tree['class']
#     if x[tree['feature']] <= tree['threshold']:
#         return predict_tree(tree['left'], x)
#     else:
#         return predict_tree(tree['right'], x)

# def random_forest_predict(X, forest):
#     y_pred = []
#     for x in X:
#         preds = [predict_tree(tree, x) for tree in forest]
#         vals, counts = np.unique(preds, return_counts=True)
#         y_pred.append(vals[np.argmax(counts)])
#     return np.array(y_pred)

# # def split_data_among_ranks(X, y, size):
# #     X_chunks = np.array_split(X, size)
# #     y_chunks = np.array_split(y, size)
# #     return X_chunks, y_chunks

# def compute_mean_std(X):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0) + 1e-9  # avoid division by zero
#     return mean, std

# def standardize(X, mean, std):
#     return (X - mean) / std

# if __name__ == "__main__":

#     if my_rank == 0:
#         np.random.seed(42)
#         n_samples = 1000
#         n_features = 5
#         X = np.random.rand(n_samples, n_features)
#         y = np.random.randint(0, 2, size=n_samples)

#         mean, std = compute_mean_std(X)

#         X = standardize(X,mean,std)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         X_train_chunks  = np.array_split(X_train, num_ranks)
#         y_train_chunks = np.array_split(y_train, num_ranks)

#         X_test_chunks  = np.array_split(X_test, num_ranks)
#         y_test_chunks = np.array_split(y_test, num_ranks)

#     else:
#         X_train_chunks, y_train_chunks = None, None
#         X_test_chunks, y_test_chunks = None, None

#     X_train_chunk = comm.scatter(X_train_chunks, root=0)
#     y_train_chunk = comm.scatter(y_train_chunks, root=0)

#     X_test_chunk = comm.scatter(X_test_chunks, root=0)
#     y_test_chunk = comm.scatter(y_test_chunks, root=0)

#     # Train forest locally
#     n_trees = 20
#     trees_per_rank = n_trees // num_ranks
#     remainder = n_trees % num_ranks
#     if my_rank < remainder:
#         trees_per_rank += 1

#     feature_subspace = int(np.sqrt(X_train_chunk.shape[1]))
#     max_depth = 6
#     min_samples_split = 5

#     local_forest = []
#     for _ in range(trees_per_rank):
#         X_sample, y_sample = bootstrap_sample(X_train_chunk, y_train_chunk)
#         tree = build_tree(X_sample, y_sample, max_depth, min_samples_split, feature_subspace)
#         local_forest.append(tree)

#     local_forest_serialized = pickle.dumps(local_forest)
#     gathered_forests = comm.gather(local_forest_serialized, root=0)

#     if my_rank == 0:
#         forest = []
#         for fs in gathered_forests:
#             forest.extend(pickle.loads(fs))

#     else:
#         forest = None

#     forest = comm.bcast(forest, root=0)

#     y_pred = random_forest_predict(X_test_chunk, forest)
#     local_correct = np.sum(y_pred == y_test_chunk)
#     local_count = len(y_pred)

#     sum_correct = comm.reduce(local_correct, op=MPI.SUM, root=0)
#     sum_count = comm.reduce(local_count, op=MPI.SUM, root=0)

#     if my_rank == 0:
#         accuracy = sum_correct/sum_count
#         print(f"Distributed Random Forest Accuracy with Standardization: {accuracy:.4f}")



################################################################################################################### SVC - not distributed

# import numpy as np

# # --- Utility Functions ---

# def train_test_split(X, y, test_size=0.2, random_seed=None):
#     if random_seed is not None:
#         np.random.seed(random_seed)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     split_idx = int(len(X) * (1 - test_size))
#     return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# def standardize_train_data(X):
#     mean = X.mean(axis=0)
#     std = X.std(axis=0)
#     std[std == 0] = 1
#     return (X - mean) / std, mean, std

# def standardize_test_data(X, mean, std):
#     return (X - mean) / std

# # --- Linear SVC using Gradient Descent (hard-margin) ---

# def svm_train(X, y, lr=0.01, lambda_param=0.01, n_iters=1000):
#     n_samples, n_features = X.shape
#     w = np.zeros(n_features)
#     b = 0

#     y_ = np.where(y <= 0, -1, 1)  # ensure binary labels are -1 or 1

#     for _ in range(n_iters):
#         for idx, x_i in enumerate(X):
#             condition = y_[idx] * (np.dot(x_i, w) + b) >= 1
#             if condition:
#                 w -= lr * (2 * lambda_param * w)
#             else:
#                 w -= lr * (2 * lambda_param * w - y_[idx] * x_i)
#                 b -= lr * y_[idx]
#     return w, b


# def svm_predict(X, w, b):
#     linear_output = np.dot(X, w) + b
#     return np.where(linear_output >= 0, 1, 0).reshape(-1, 1)

# # --- Example usage ---

# # Sample binary classification data
# np.random.seed(0)
# X = np.random.randn(500, 5)
# y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)  # simple separable target

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

# # Standardize
# X_train_std, mean, std = standardize_train_data(X_train)
# X_test_std = standardize_test_data(X_test, mean, std)

# # Train SVM
# w, b = svm_train(X_train_std, y_train, lr=0.001, lambda_param=0.01, n_iters=1000)

# # Predict
# y_pred = svm_predict(X_test_std, w, b)

# # Accuracy
# accuracy = np.mean(y_pred == y_test)
# print(f"SVC Accuracy: {accuracy:.4f}")


########################################################################################################################## SVC distributed

# from mpi4py import MPI
# import numpy as np

# # --- MPI Section ---

# comm = MPI.COMM_WORLD
# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# def train_test_split(X, y, test_size=0.2, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     indices = np.random.permutation(len(X))
#     split = int(len(X) * (1 - test_size))
#     return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

# def standardize_train_data(X):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     std[std == 0] = 1.0
#     X_std = (X - mean) / std
#     return X_std, mean, std

# def standardize_test_data(X, mean, std):
#     return (X - mean) / std

# def local_svm_train(X, y, lr=0.01, lambda_param=0.01, n_iters=10000):
#     n_samples, n_features = X.shape
#     tolerance = 1e-7
#     w = np.zeros(n_features)
#     b = 0
#     y_ = np.where(y <= 0, -1, 1)

#     for iter in range(n_iters):
#         for idx, x_i in enumerate(X):
#             condition = y_[idx] * (np.dot(x_i, w) + b) >= 1
#             if condition:
#                 new_w = (w - lr * (2 * lambda_param * w))
#             else:
#                 new_w = w - (lr * (2 * lambda_param * w - y_[idx] * x_i))
#                 new_b = b- (lr * y_[idx])

#             if np.all(np.abs(w-new_w) < tolerance):
#                 print(f"rank {my_rank} converged after {iter} iterations")
#                 w = new_w
#                 b = new_b
#                 break
#             w = new_w
#             b = new_b

#     return w, b

# def svm_predict(X, w, b):
#     linear_output = np.dot(X, w) + b
#     return np.where(linear_output >= 0, 1, 0).reshape(-1, 1)



# # Master process
# if my_rank == 0:
#     np.random.seed(0)
#     X = np.random.randn(500, 5)
#     y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

#     X_train, mean, std = standardize_train_data(X_train)
#     X_test = standardize_test_data(X_test, mean, std)

#     X_train_chunks = np.array_split(X_train,num_ranks)
#     X_test_chunks = np.array_split(X_test,num_ranks)

#     y_train_chunks = np.array_split(y_train,num_ranks)
#     y_test_chunks = np.array_split(y_test,num_ranks)

# else:
#     X_train_chunks = None
#     y_train_chunks = None
#     X_test_chunks = None
#     y_test_chunks = None
#     mean = None
#     std = None

# X_train_chunk = comm.scatter(X_train_chunks, root=0)
# y_train_chunk = comm.scatter(y_train_chunks, root=0)

# X_test_chunk = comm.scatter(X_test_chunks, root=0)
# y_test_chunk = comm.scatter(y_test_chunks, root=0)

# # Local training
# w_local, b_local = local_svm_train(X_train_chunk, y_train_chunk, lr=0.001, lambda_param=0.01, n_iters=10000)

# # Aggregate weights
# # w_global = np.zeros_like(w_local)
# # b_global = np.zeros(1)

# w_global = comm.reduce(w_local, op=MPI.SUM, root=0)
# b_global = comm.reduce(b_local, op=MPI.SUM, root = 0)

# # Average
# if my_rank == 0:
#     w_global /= num_ranks
#     b_global /= num_ranks

# else:
#     w_global = None
#     b_global = None

# b_global = comm.bcast(b_global,root=0)
# w_global = comm.bcast(w_global, root=0)

# y_pred = svm_predict(X_test_chunk, w_global, b_global[0])
# local_correct = np.sum(y_pred == y_test_chunk)
# local_count = len(y_pred)

# total_correct = comm.reduce(local_correct, op=MPI.SUM, root=0)
# total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# if my_rank == 0:
#     acc = total_correct/total_count
#     print(f"Distributed SVC Accuracy: {acc:.4f}")


####################################################################################################################### SVR Non-distributed

# import numpy as np

# def standardize_data(X):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     return (X - mean) / std, mean, std

# def svr_train(X, y, lr=0.001, lambda_param=0.01, epsilon=0.1, n_iters=1000):
#     n_samples, n_features = X.shape
#     w = np.zeros(n_features)
#     b = 0

#     for _ in range(n_iters):
#         for i in range(n_samples):
#             x_i = X[i]
#             y_i = y[i]
#             y_pred = np.dot(w, x_i) + b
#             error = y_pred - y_i

#             if error > epsilon:
#                 grad_w = lambda_param * w + x_i
#                 grad_b = 1
#             elif error < -epsilon:
#                 grad_w = lambda_param * w - x_i
#                 grad_b = -1
#             else:
#                 grad_w = lambda_param * w
#                 grad_b = 0

#             w -= lr * grad_w
#             b -= lr * grad_b

#     return w, b

# def svr_predict(X, w, b):
#     return np.dot(X, w) + b

# # Example usage:
# np.random.seed(0)
# X = np.random.rand(100, 5)
# y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1  # Linear + noise

# # Split manually
# split = int(0.8 * len(X))
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# # Standardize
# X_train_std, mean, std = standardize_data(X_train)
# X_test_std = (X_test - mean) / std

# # Train
# w, b = svr_train(X_train_std, y_train)

# # Predict
# predictions = svr_predict(X_test_std, w, b)
# mse = np.mean((predictions - y_test) ** 2)
# print("MSE:", mse)


##################################################################################################################### SVR - distributed

# from mpi4py import MPI
# import numpy as np

# # Initialize MPI
# comm = MPI.COMM_WORLD
# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# # Manual standardization functions
# def standardize_train_data(X):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     std[std == 0] = 1.0
#     X_std = (X - mean) / std
#     return X_std, mean, std

# def standardize_test_data(X, mean, std):
#     return (X - mean) / std

# def train_test_split(X, y, test_size=0.2, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     indices = np.random.permutation(len(X))
#     split = int(len(X) * (1 - test_size))
#     return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]


# # SVR training function
# def svr_train(X, y, lr=0.01, C=1.0, epsilon=0.1, n_iters=50000):
#     n_samples, n_features = X.shape
#     w = np.zeros(n_features)
#     b = 0.0

#     for _ in range(n_iters):
#         for i in range(n_samples):
#             xi = X[i]
#             yi = y[i]
#             y_pred = np.dot(xi, w) + b
#             error = y_pred - yi

#             if abs(error) > epsilon:
#                 grad_w = np.sign(error) * xi + 2 * C * w
#                 grad_b = np.sign(error)
#                 w -= lr * grad_w
#                 b -= lr * grad_b

#     return w, b

# # SVR prediction
# def svr_predict(X, w, b):
#     return np.dot(X, w) + b

# # Data generation and distribution
# if my_rank == 0:
#     # Create synthetic data
#     np.random.seed(42)
#     X = np.random.rand(1000, 5)
#     true_w = np.array([2, -1, 3, 0.5, -2])
#     y = X @ true_w + np.random.normal(0, 0.5, 1000)
    

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

#     X_train, mean, std = standardize_train_data(X_train)
#     X_test = standardize_test_data(X_test, mean, std)

#     X_train_chunks = np.array_split(X_train,num_ranks)
#     X_test_chunks = np.array_split(X_test,num_ranks)

#     y_train_chunks = np.array_split(y_train,num_ranks)
#     y_test_chunks = np.array_split(y_test,num_ranks)
    
# else:
#     X_train_chunks = None
#     y_train_chunks = None
#     X_test_chunks = None
#     y_test_chunks = None
    

# # Distribute data
# X_train_chunk = comm.scatter(X_train_chunks, root=0)
# y_train_chunk = comm.scatter(y_train_chunks, root=0)
# X_test_chunk = comm.scatter(X_test_chunks, root=0)
# y_test_chunk = comm.scatter(y_test_chunks, root=0)

# # Train local model
# w_local, b_local = svr_train(X_train_chunk, y_train_chunk, lr=0.01, C=1.0, epsilon=0.1, n_iters=50000)

# # Gather weights and biases
# w_global = comm.gather(w_local, root=0)
# b_global = comm.gather(b_local, root=0)

# # Evaluation at root
# if my_rank == 0:
#     w_avg = np.mean(w_global, axis=0)
#     b_avg = np.mean(b_global)

# else:
#     w_avg = None
#     b_avg = None

# w_avg = comm.bcast(w_avg, root=0)
# b_avg = comm.bcast(b_avg, root=0)

# y_pred = svr_predict(X_test_chunk, w_avg, b_avg)
# # rmse = np.sqrt(np.mean((y_test_chunk - y_pred) ** 2))
# # r2 = 1 - np.sum((y_test_chunk - y_pred) ** 2) / np.sum((y_test_chunk - np.mean(y_test_chunk)) ** 2)

# # Local error metrics
# local_sse = np.sum((y_test_chunk - y_pred) ** 2)
# local_count = len(y_test_chunk)
# local_sum_y = np.sum(y_test_chunk)
# local_sum_y_squared = np.sum(y_test_chunk ** 2)

# # Gather components at root
# total_sse = comm.reduce(local_sse, op=MPI.SUM, root=0)
# total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
# sum_y = comm.reduce(local_sum_y, op=MPI.SUM, root=0)
# sum_y_squared = comm.reduce(local_sum_y_squared, op=MPI.SUM, root=0)

# # Final evaluation at root
# if my_rank == 0:
#     y_mean_global = sum_y / total_count
#     tss = sum_y_squared - total_count * y_mean_global**2

#     rmse_global = np.sqrt(total_sse / total_count)
#     r2_global = 1 - (total_sse / tss)

#     print(f"\nGlobal RMSE: {rmse_global:.4f}")
#     print(f"Global r2: {r2_global:.4f}")


################################################################################################################## One-hot encoding

# def manual_label_encode(feature_column):
#     unique_categories = list(set(feature_column))
#     mapping = {category: idx for idx, category in enumerate(unique_categories)}
#     encoded = [mapping[value] for value in feature_column]
#     return np.array(encoded), mapping


###################################################################################################################### MapReduce
############################### First mapper
# import sys
# import re

# window_length = 4

# punctuations = r'[.,!?;:\-_—()[\]{}\'"…`‘’“”/\\|@#$%^&*~+=<>-]'

# stopwords = set([
#     'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
#     'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
#     'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 
#     'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 
#     'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', 
#     "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 
#     'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", 
#     "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 
#     'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 
#     'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 
#     'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', 
#     "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 
#     'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', 
#     "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 
#     'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", 
#     "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', 
#     "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', 
#     "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 
#     'your', 'yours', 'yourself', 'yourselves'
# ])


# for line in sys.stdin:
#     if not line:
#         continue
#     line = line.strip()
    
#     cleaned_words = []
#     words = line.lower().split()

#     for word in words:
#         word = re.sub(punctuations, '', word)

#         if word and word not in stopwords:
#             cleaned_words.append(word)

#     for i in range(len(cleaned_words)):
#         current_word = cleaned_words[i]
#         linked_words = cleaned_words[i+1:i+window_length]

#         for linked_word in linked_words:
#             if current_word != linked_word:
#                 print(f"{current_word}\t1")
#                 print(f"{linked_word}\t1")

# ########################################### First reducer

# import sys

# current_word = None
# current_links = 0
# threshold = 100

# for line in sys.stdin:
#     if not line:
#         continue
#     line = line.strip()

#     word, links = line.split("\t")
#     word = word.strip()
#     links = int(links)

#     if word == current_word:
#         current_links += links

#     else:
#         if current_word is not None and current_links > threshold:
#             print(f"{current_word}\t{current_links}")

#         current_word = word
#         current_links = links

# if current_word is not None and current_links > threshold:
#     print(f"{current_word}\t{current_links}")


# ####################################### Second mapper

# import sys

# for line in sys.stdin:
#     if not line:
#         continue
#     line = line.strip()

#     word, links = line.split("\t")
#     links = int(links)

#     inv_links = 100000 - links

#     print(f"{inv_links}\t{word}")





# ################################# Second reducer

# import sys

# for line in sys.stdin:
#     if not line:
#         continue
#     line = line.strip()

#     inv_links, word = line.split("\t")
#     inv_links = int(inv_links)
#     links = 100000 - inv_links

#     print(f"{word}\t{links}")



##########################################################################################################################3### PCA

# from mpi4py import MPI
# import numpy as np

# def compute_local_covariance(X_local):
#     """Compute the covariance matrix for local data chunk."""
#     X_centered = X_local - np.mean(X_local, axis=0)
#     cov_local = np.dot(X_centered.T, X_centered) / (X_local.shape[0] - 1)
#     return cov_local, X_centered.shape[0]


# comm = MPI.COMM_WORLD
# my_rank = comm.Get_rank()
# num_ranks = comm.Get_size()

# # Define data size (only on rank 0)
# n_samples = 1000
# n_features = 10
# data = None

# if my_rank == 0:
#     # Generate synthetic dataset
#     np.random.seed(42)
#     data = np.random.rand(n_samples, n_features)

#     # Split data equally among all processes
#     chunks = np.array_split(data, num_ranks, axis=0)
# else:
#     chunks = None

# # Scatter data chunks to all processes
# local_data = comm.scatter(chunks, root=0)

# # Compute local covariance matrix and number of samples
# local_cov, local_n = compute_local_covariance(local_data)

# # Gather local covariance matrices and sample counts at root
# covs = comm.gather(local_cov, root=0)
# counts = comm.gather(local_n, root=0)
# if my_rank == 0:
#     # Compute weighted average of covariance matrices
#     total_samples = sum(counts)
#     global_cov = sum(covs[i] * counts[i] for i in range(num_ranks)) / total_samples

#     # Perform eigen decomposition
#     eig_vals, eig_vecs = np.linalg.eigh(global_cov)

#     # Sort eigenvalues and eigenvectors in descending order
#     idx = np.argsort(eig_vals)[::-1]
#     eig_vals = eig_vals[idx]
#     eig_vecs = eig_vecs[:, idx]

#     print("Top 3 principal components (eigenvectors):")
#     print(eig_vecs[:, :3])
#     print("Top 3 eigenvalues:")
#     print(eig_vals[:3])


######################################################################################################### Distributed Decision trees

# from mpi4py import MPI
# import numpy as np

# def gini_impurity(y):
#     """Calculate Gini impurity of a label array."""
#     classes, counts = np.unique(y, return_counts=True)
#     probs = counts / len(y)
#     return 1 - np.sum(probs ** 2)

# def best_split(X, y):
#     """Find the best feature and threshold to split on."""
#     n_samples, n_features = X.shape
#     best_gini = 1
#     best_feature = None
#     best_threshold = None

#     for feature in range(n_features):
#         thresholds = np.unique(X[:, feature])
#         for t in thresholds:
#             left_mask = X[:, feature] <= t
#             right_mask = ~left_mask

#             if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
#                 continue

#             gini_left = gini_impurity(y[left_mask])
#             gini_right = gini_impurity(y[right_mask])

#             weighted_gini = (len(y[left_mask]) * gini_left + len(y[right_mask]) * gini_right) / len(y)

#             if weighted_gini < best_gini:
#                 best_gini = weighted_gini
#                 best_feature = feature
#                 best_threshold = t

#     return best_feature, best_threshold, best_gini


# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Generate synthetic data (only on rank 0)
# if rank == 0:
#     from sklearn.datasets import make_classification
#     X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42)
#     chunks_X = np.array_split(X, size)
#     chunks_y = np.array_split(y, size)
# else:
#     chunks_X = None
#     chunks_y = None

# # Scatter data to processes
# X_local = comm.scatter(chunks_X, root=0)
# y_local = comm.scatter(chunks_y, root=0)

# # Each process computes the best local split
# local_result = best_split(X_local, y_local)

# # Gather all local split results
# all_results = comm.gather(local_result, root=0)

# if rank == 0:
#     # Choose best global split
#     best = None
#     for res in all_results:
#         if res[0] is not None:
#             if best is None or res[2] < best[2]:
#                 best = res

#     print("Best global split found:")
#     print(f"Feature: {best[0]}, Threshold: {best[1]}, Gini: {best[2]}")


#######################################################################################################
# from mpi4py import MPI
# import numpy as np
# from scipy.spatial.distance import euclidean
# import sys

# def compute_pairwise_distances_chunk(data, start, end):
#     n = len(data)
#     distances = []

#     for i in range(start, end):
#         for j in range(i + 1, n):
#             dist = euclidean(data[i], data[j])
#             distances.append((i, j, dist))

#     return distances

# def find_clusters_to_merge(distances, active_clusters):
#     min_dist = float('inf')
#     pair = (-1, -1)

#     for i, j, d in distances:
#         if i in active_clusters and j in active_clusters and d < min_dist:
#             min_dist = d
#             pair = (i, j)

#     return pair

# def hierarchical_clustering(data):
#     n = len(data)
#     clusters = {i: [i] for i in range(n)}
#     active_clusters = set(clusters.keys())
#     merge_history = []

#     while len(active_clusters) > 1:
#         # Compute pairwise distances between all active clusters
#         distances = []
#         for i in active_clusters:
#             for j in active_clusters:
#                 if i < j:
#                     min_dist = min(
#                         euclidean(data[p1], data[p2])
#                         for p1 in clusters[i]
#                         for p2 in clusters[j]
#                     )
#                     distances.append((i, j, min_dist))

#         # Find the pair to merge
#         i, j = find_clusters_to_merge(distances, active_clusters)

#         # Merge clusters
#         new_cluster_id = max(clusters.keys()) + 1
#         clusters[new_cluster_id] = clusters[i] + clusters[j]
#         merge_history.append((i, j))
#         active_clusters.remove(i)
#         active_clusters.remove(j)
#         active_clusters.add(new_cluster_id)

#     return merge_history

# def main():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     # Sample data (e.g., 2D points)
#     if rank == 0:
#         data = np.array([
#             [1.0, 2.0],
#             [1.5, 1.8],
#             [5.0, 8.0],
#             [8.0, 8.0],
#             [1.0, 0.6],
#             [9.0, 11.0]
#         ])
#     else:
#         data = None

#     # Broadcast data to all processes
#     data = comm.bcast(data, root=0)
#     n = len(data)

#     # Each process computes a chunk of pairwise distances
#     total_pairs = n * (n - 1) // 2
#     pairs_per_proc = total_pairs // size
#     remainder = total_pairs % size

#     start_index = sum(
#         n - 1 - i for i in range(rank * (n // size))
#     ) + min(rank, remainder)
#     end_index = start_index + pairs_per_proc + (1 if rank < remainder else 0)

#     chunk_distances = compute_pairwise_distances_chunk(data, 0, n)
#     gathered = comm.gather(chunk_distances, root=0)

#     if rank == 0:
#         all_distances = [d for sublist in gathered for d in sublist]
#         print("Pairwise distances computed across all processes.")

#         # Perform hierarchical clustering (on master)
#         merge_history = hierarchical_clustering(data)
#         print("Merge history (cluster indices):", merge_history)

# if __name__ == "__main__":
#     main()


#######################################################################################################

# from mpi4py import MPI
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import euclidean

# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Load data (only rank 0 loads and broadcasts it)
# if rank == 0:
#     df1 = pd.read_csv("hierarical_clustering_data_4.csv")
#     df2 = pd.read_csv("hierarical_clustering_data_5.csv")
#     df3 = pd.read_csv("hierarical_clustering_data_6.csv")
#     data = pd.concat([df1, df2, df3]).drop_duplicates().values
# else:
#     data = None

# # Broadcast the data
# data = comm.bcast(data, root=0)

# # Each point starts in its own cluster
# clusters = [{i} for i in range(len(data))]

# def pairwise_distances(clusters, data):
#     """Compute distances between all pairs of clusters"""
#     pairs = []
#     for i in range(len(clusters)):
#         for j in range(i+1, len(clusters)):
#             # single linkage
#             min_dist = float('inf')
#             for idx1 in clusters[i]:
#                 for idx2 in clusters[j]:
#                     dist = euclidean(data[idx1], data[idx2])
#                     if dist < min_dist:
#                         min_dist = dist
#             pairs.append(((i, j), min_dist))
#     return pairs

# def parallel_min_distance(pairs):
#     """Find minimum pair using MPI reduction"""
#     # Split pairs among processes
#     chunk = np.array_split(pairs, size)[rank]
#     local_min = min(chunk, key=lambda x: x[1]) if chunk else ((-1, -1), float('inf'))
#     global_min = comm.allreduce(local_min, op=MPI.MIN)
#     return global_min

# # Main clustering loop
# while len(clusters) > 1:
#     pairs = pairwise_distances(clusters, data)
#     (i, j), min_dist = parallel_min_distance(pairs)

#     if rank == 0:
#         # Merge clusters
#         new_cluster = clusters[i].union(clusters[j])
#         clusters.pop(j)
#         clusters.pop(i)
#         clusters.insert(0, new_cluster)

#     clusters = comm.bcast(clusters, root=0)

# # Final result
# if rank == 0:
#     print("Final cluster contains all points.")












    







