# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# iterations = 20
# x = 3.5
# learning_rate = 0.1

# def f(x):
#     return x**2

# def gradient(x):
#     return 2*x

# x_list = [x]
# for iteration in range(iterations):
#     x =  x - (learning_rate*gradient(x))
#     x_list.append(x)

# print(x_list)

# x_plot = np.linspace(-4,4,100)
# y_plot = [f(i) for i in x_plot]

# plt.scatter(x_list, [f(i) for i in x_list], color='red', label='Gradient Descent Steps')
# plt.plot(x_plot, y_plot, label='f(x) = x^2')
# plt.grid(True)
# plt.title("A function x^2 and its gradient descent")
# plt.legend()
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# x = np.arange(0,1,0.01)
# y = x + np.random.normal(0,0.2,len(x))

# # Initialize the parameters
# theta1 = -0.5
# theta2 = 0.2
# learning_rate = 0.006
# max_iterations = 20
# tolerance = 1e-6

# def y_hat(x, theta1, theta2):
#     return theta1*x + theta2

# y_opt = y_hat(x,1,0)

# def gradient_descent(x,y,theta1,theta2,learning_rate):
#     n_samples = len(x)

#     y_pred = y_hat(x, theta1, theta2)

#     theta1_grad = (-2/n_samples)*np.sum(x*(y-y_pred))
#     theta2_grad = (-2/n_samples)*np.sum((y-y_pred))

#     theta1 -= learning_rate*theta1_grad
#     theta2 -= learning_rate*theta2_grad

#     return theta1, theta2


# def mse(x,y,theta1,theta2):
#     y_pred = y_hat(x,theta1,theta2)
#     return np.mean((y-y_pred)**2)

# def fit(x, y, theta1, theta2, learning_rate, max_iterations, tolerance):
#     loss_list = []
#     prev_loss = np.inf
#     for iteration in range(max_iterations):
#         theta1,theta2 = gradient_descent(x,y,theta1,theta2,learning_rate)
#         current_loss = mse(x,y,theta1,theta2)
#         loss_list.append(current_loss)

#         print(f"Epoch {iteration+1}/{max_iterations}: theta1: {theta1}, theta2: {theta2}, loss: {current_loss}")
#         if abs(prev_loss-current_loss) < tolerance:
#             print(f"Converged after {iteration} iterations")
#             break
#         prev_loss = current_loss
#     return theta1, theta2

# theta1,theta2 = fit(x, y, theta1, theta2, learning_rate, max_iterations, tolerance)

# y_pred = y_hat(x,theta1,theta2)
# plt.scatter(x,y,label='Data')
# plt.plot(x, y_opt,color='red',label='Optimal')
# plt.plot(x, y_pred,color='blue',label='Predicted')
# plt.legend()
# plt.show()

from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_ranks = comm.Get_size()


max_iterations = 12000

if my_rank == 0:
    X = np.arange(0,1,0.01)
    y = X + np.random.normal(0,0.2,len(X))
    learning_rate = 0.006

    chunk_size = len(X)//num_ranks
    remainders = len(X)%num_ranks

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

theta1 = comm.bcast(theta1, root=0)
theta2 = comm.bcast(theta2, root=0)
learning_rate = comm.bcast(learning_rate, root=0)

def y_hat(theta1, theta2, X):
    return (theta1*X)+theta2

def gradient_descent(X,y,theta1,theta2):
    n_samples = len(X)
    y_pred = y_hat(theta1, theta2, X)

    theta1_grad = (-2/n_samples)*np.sum(X*(y-y_pred))
    theta2_grad = (-2/n_samples)*np.sum((y-y_pred))

    # new_theta1 = theta1 - theta1_grad
    # new_theta2 = theta2 - theta2_grad

    return theta1_grad, theta2_grad

X_chunk = comm.scatter(X_chunks, root=0)
y_chunk = comm.scatter(y_chunks, root=0)

for iteration in range(max_iterations):
    theta1_grad_local, theta2_grad_local = gradient_descent(X_chunk, y_chunk, theta1,theta2)

    theta1_grad_global = comm.reduce(theta1_grad_local, op=MPI.SUM, root=0)
    theta2_grad_global = comm.reduce(theta2_grad_local, op=MPI.SUM, root=0)

    if my_rank == 0:
        theta1 -= learning_rate*(theta1_grad_global/num_ranks)
        theta2 -= learning_rate*(theta2_grad_global/num_ranks)

        print(f"Epoch: {iteration+1}/{max_iterations}, theta1: {theta1}, theta2: {theta2}")


    theta1 = comm.bcast(theta1, root=0)
    theta2 = comm.bcast(theta2, root=0)

if my_rank == 0:
    y_opt = y_hat(1,0,X)
    y_pred = y_hat(theta1,theta2,X)
    plt.scatter(X,y,label='Data')
    plt.plot(X, y_opt,color='red',label='Optimal')
    plt.plot(X, y_pred,color='blue',label='Predicted')
    plt.legend()
    plt.show()










