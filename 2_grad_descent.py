from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank() # ID of rank being called
num_ranks = comm.Get_size() # num of ranks

max_iterations = 50000 # maximum iterations
tolerance = 1e-20
# Optimal values of theta1 and theta2
theta1_opt = 1.0
theta2_opt = 0.0

if my_rank == 0:
    # Synthetic dataset
    X = np.arange(0,1,0.01)
    y = X + np.random.normal(0,0.2,len(X))
    learning_rate = 0.006

    # Split the X and y into chunks to be distributed
    X_chunks = np.array_split(X, num_ranks)
    y_chunks = np.array_split(y, num_ranks)

    # Initializa theta1 and theta2
    theta1 = np.random.normal(0,1)
    theta2 = np.random.normal(0,1)

else:
    X_chunks = None
    y_chunks = None
    theta1 = None
    theta2 = None
    learning_rate = None

# Broadcast the parameters
theta1 = comm.bcast(theta1, root=0)
theta2 = comm.bcast(theta2, root=0)
learning_rate = comm.bcast(learning_rate, root=0)

# Function for prediction
def y_hat(theta1, theta2, X):
    return (theta1*X)+theta2

# Function to calculate the gradients which will be used to update the thetas
def gradient_descent(X,y,theta1,theta2):
    n_samples = len(X)
    y_pred = y_hat(theta1, theta2, X) # Make predictions

    theta1_grad = (-2/n_samples)*np.sum(X*(y-y_pred))
    theta2_grad = (-2/n_samples)*np.sum((y-y_pred))
    return theta1_grad, theta2_grad

# Distribute the data across all processes
X_chunk = comm.scatter(X_chunks, root=0)
y_chunk = comm.scatter(y_chunks, root=0)

for iteration in range(max_iterations):
    # Calculate gradients
    theta1_grad_local, theta2_grad_local = gradient_descent(X_chunk, y_chunk, theta1,theta2)

    # Sum the gradient across all processes
    theta1_grad_global = comm.reduce(theta1_grad_local, op=MPI.SUM, root=0)
    theta2_grad_global = comm.reduce(theta2_grad_local, op=MPI.SUM, root=0)

    if my_rank == 0:
        # Update the parameters using the average gradient across all processes
        theta1 -= learning_rate*(theta1_grad_global)
        theta2 -= learning_rate*(theta2_grad_global)
        # print(f"Epoch: {iteration+1}/{max_iterations}, theta1: {theta1}, theta2: {theta2}")
        # Checking how many iterations it takes to reach optimal parameters
        converged = theta1_opt - theta1< tolerance and theta2 - theta2_opt < tolerance
        # converged = theta1 == theta1_opt and theta2 == theta2_opt

    else:
        converged = None
        theta1 = None
        theta2 = None

    converged = comm.bcast(converged, root=0)
    theta1 = comm.bcast(theta1, root=0)
    theta2 = comm.bcast(theta2, root=0)

    if my_rank == 0:
        if converged:
            # If it converges, print these
            print(f"Converged to the nearest theta1 optimal value of 1.0 and nearest theta2 optimal value of 0.0 after {iteration} iterations")
            print(f"Epoch: {iteration+1}/{max_iterations}, theta1: {theta1}, theta2: {theta2}")
            break

    

if my_rank == 0:
    # Plots to show results
    y_opt = y_hat(1,0,X)
    y_pred = y_hat(theta1,theta2,X)
    plt.scatter(X,y,label='Data')
    plt.plot(X, y_opt,color='red',label='Optimal')
    plt.plot(X, y_pred,color='blue',label='Predicted')
    plt.legend()
    plt.show()