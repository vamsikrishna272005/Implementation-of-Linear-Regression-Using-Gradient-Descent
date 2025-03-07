# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation:
    * Read the dataset from a CSV file using pandas.
    * Separate the features (X) and the target variable (y) from the dataset.
    * Convert these into NumPy arrays and scale them using StandardScaler to ensure all features have a mean of 0 and a standard deviation of 1, which helps in the convergence of gradient descent.

2. Feature Engineering:
    * Add an intercept term (a column of ones) to the feature matrix X. This is necessary for the bias (intercept) in the linear regression model.

3. Initialization:
    * Initialize the weights (theta) to zeros. The number of weights is equal to the number of features plus one (for the intercept).

4. Gradient Descent:
    * Iterate a specified number of times (num_iters).

    *  In each iteration:
        * Compute the predictions by multiplying the feature matrix X with the weights theta.
        * Calculate the errors by subtracting the actual target values (y) from the predictions.
        * Update the weights (theta) using the gradient descent update rule:

        $$
        \theta = \alpha\frac{1}{m}X^T(X\theta - y)
        $$

        where:
        * α is the learning rate.
        * m is the number of training examples.

5. Prediction:
    * To predict the target value for new data, scale the new data using the same scaler.
    * Use the learned weights (theta) to make predictions on the scaled new data.

## Program:
Program to implement the linear regression using gradient descent.

Developed by: Vamsi Krishna G

RegisterNumber: 212223220120

```python
import numpy as np
import pandas as pd
from sklearn. preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
  # Add a column of ones to x for the intercept term
  X=np.c_[np.ones(len(X1)), X1]

  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)

  # Perform gradient descent
  for _ in range(num_iters):
    # Calculate predictions
    predictions = (X).dot(theta).reshape(-1, 1)

  # Calculate errors
  errors = (predictions - y).reshape(-1,1)

  # Update theta using gradient descent
  theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

  return theta
data = pd.read_csv('50_Startups.csv',header=None)

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'x'
X = (data.iloc[1:, :- 2].values)
X1=X.astype(float)
scaler = StandardScaler()
y= (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
# Example usage
#X= np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
#y = np.array([2, 7, 11, 16])

# Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/ab276837-f1d2-42c4-9148-00e62553b71d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
