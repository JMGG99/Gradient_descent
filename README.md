# Linear Regression with Gradient Descent

## Overview
This project implements a simple linear regression model using gradient descent. The model aims to find the best-fit line for a set of generated data points following the equation:

\[ y = w \cdot x + b \]

where:
- `w` is the weight (slope),
- `b` is the bias (intercept),
- `x` is the input feature, and
- `y` is the predicted output.

## Features
- Generates synthetic data with noise.
- Implements a linear model.
- Uses mean squared error (MSE) as the loss function.
- Optimizes parameters `w` and `b` using gradient descent.
- Visualizes the data, loss reduction over epochs, and the final regression line.

## Installation
Ensure you have Python installed along with the required libraries:

```bash
pip install numpy matplotlib
```

## Usage
Run the script to:
1. Generate synthetic data.
2. Train the model using gradient descent.
3. Plot the training loss over epochs.
4. Display the final regression line.

Execute the script:
```bash
python script_name.py
```

## Functions
- `generate_data()`: Generates `x` values and corresponding `y` values with added noise.
- `model(x, w, b)`: Computes predictions using the linear equation.
- `compute_loss(ys_pred, ys)`: Calculates the mean squared error.
- `gradient_descent(xs, ys, w, b, learning_rate)`: Updates `w` and `b` using gradient descent.
- `train(xs, ys, w, b, learning_rate, epochs)`: Trains the model and tracks the loss over iterations.
- `plot_loss(loss_history)`: Plots loss over training epochs.
- `plot_regression_line(xs, ys, w, b)`: Displays the data points and the fitted regression line.

## Results
The script trains the model and adjusts `w` and `b` to minimize the loss function. After training, it plots the regression line alongside the original data points.


