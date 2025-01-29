import numpy as np
import matplotlib.pyplot as plt

# Function to generate data
def generate_data():
    xs = np.linspace(1, 20, 20)
    ys = 3 * xs + 5 + np.random.normal(0, 2, len(xs))
    return xs, ys

# Function to plot data
def plot_data(xs, ys):
    plt.scatter(xs, ys)
    plt.title("Data Visualization")
    plt.xlabel("xs")
    plt.ylabel("ys")
    plt.show()

# Linear Regression Model: y = w * x + b
def model(x, w, b):
    return w * x + b

# Loss Function: Mean Squared Error (MSE)
def compute_loss(ys_pred, ys):
    return np.mean((ys_pred - ys) ** 2)

# Gradient Descent: Update parameters to minimize loss
def gradient_descent(xs, ys, w, b, learning_rate):
    N = len(xs)  # Number of data points
    ys_pred = model(xs, w, b)  # Predicted values using current model
    loss = compute_loss(ys_pred, ys)  # Compute the loss
    
    # Gradients of the loss with respect to w and b
    dw = -2 * np.sum(xs * (ys - ys_pred)) / N
    db = -2 * np.sum(ys - ys_pred) / N
    
    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db
    
    return w, b, loss  # Return updated parameters and current loss

# Training the model with gradient descent
def train(xs, ys, w, b, learning_rate, epochs):
    loss_history = []  # To track the loss for each epoch
    for epoch in range(epochs):
        w, b, loss = gradient_descent(xs, ys, w, b, learning_rate)  # Update weights and bias
        loss_history.append(loss)  # Store loss for this epoch
        
        if epoch % 100 == 0:  # Print every 100 epochs
            print(f"Epoch {epoch}: w = {w:.4f}, b = {b:.4f}, Loss = {loss:.4f}")
    
    return w, b, loss_history  # Return final parameters and loss history

# Visualize the loss over epochs
def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

# Final plot of the data with the regression line
def plot_regression_line(xs, ys, w, b):
    plt.scatter(xs, ys, label="Data points")  # Plot the actual data points
    plt.plot(xs, model(xs, w, b), color='red', label="Regression line")  # Plot the fitted line
    plt.title("Linear Regression Model")
    plt.xlabel("xs")
    plt.ylabel("ys")
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    xs, ys = generate_data()  # Get data
    
    # Initialize parameters
    w = np.random.randn()  # Random initial weight
    b = np.random.randn()  # Random initial bias
    learning_rate = 0.001
    epochs = 1000
    
    # Train the model
    w, b, loss_history = train(xs, ys, w, b, learning_rate, epochs)
    
    # Plot the loss over training
    plot_loss(loss_history)
    
    # Final plot of the data with the regression line
    plot_regression_line(xs, ys, w, b)
