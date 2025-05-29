import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Synthetic dataset
x = np.arange(0,1,0.01)
y = x + np.random.normal(0,0.2,len(x))

# Define the parameters
theta1 = -0.5
theta2 = 0.2
learning_rate = 0.006
max_iterations = 20

# Function to make predictions
def y_hat(x, theta1, theta2):
    return theta1*x + theta2

# To calculate optimal predictions using theta1=1.0 and theta2=0.0
y_opt = y_hat(x,1,0)

# Implementing the gradient descent algorithm
def gradient_descent(x,y,theta1,theta2,learning_rate):
    # num of samples for calcating average
    n_samples = len(x)
    # Prediction
    y_pred = y_hat(x, theta1, theta2)

    # Calculate the gradients as the first derivation of the loss function with respect to theta1 and theta2
    theta1_grad = (-2/n_samples)*np.sum(x*(y-y_pred))
    theta2_grad = (-2/n_samples)*np.sum((y-y_pred))

    # Updating the weights
    theta1 -= learning_rate*theta1_grad
    theta2 -= learning_rate*theta2_grad

    return theta1, theta2

# Function to calculate the mean squared error
def mse(x,y,theta1,theta2):
    y_pred = y_hat(x,theta1,theta2)
    return np.mean((y-y_pred)**2)

# Training to find optimal parameters after maximum iterations
def fit(x, y, theta1, theta2, learning_rate, max_iterations):
    history = []
    prev_loss = np.inf # Initialize mse
    # Iterate
    for iteration in range(max_iterations):
        # Update the parameters using gradient_descent function
        theta1,theta2 = gradient_descent(x,y,theta1,theta2,learning_rate)
        # Calculate the mse for this iteration
        current_loss = mse(x,y,theta1,theta2)
        # Add it to the history
        history.append(current_loss)

        print(f"Epoch {iteration+1}/{max_iterations}: theta1: {theta1}, theta2: {theta2}, loss: {current_loss}")
    return theta1, theta2, history

theta1,theta2,history = fit(x, y, theta1, theta2, learning_rate, max_iterations)

# Make predictions using the parameters after the 20th iteration
y_pred = y_hat(x,theta1,theta2)
# Plot the results
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# Plot the original data
axs[0].scatter(x,y,label='Data')
# Plot the results of the optimal model
axs[0].plot(x, y_opt,color='red',label='Optimal')
# Plot the results using 20 iteration parameters
axs[0].plot(x, y_pred,color='blue',label='Predicted')
# Plot the mse history
axs[1].scatter([1+i for i in range(max_iterations)], history, label='history')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()