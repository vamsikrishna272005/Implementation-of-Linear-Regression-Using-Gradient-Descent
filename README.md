# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VAMSI KRISHNA G
RegisterNumber:  212223220120
*/
```
```python 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

### data information

![image](https://github.com/HIRU-VIRU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145972122/62833b13-f0ee-48f0-8d66-67635e4404e6)


### Value of x
![image](https://github.com/HIRU-VIRU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145972122/a026977b-70de-475a-bf0d-65918be7158d)


### Value of X1_scaled

![image](https://github.com/HIRU-VIRU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145972122/ee49a231-be1b-47f9-8d73-18fd9c7ca74c)


### predicted value

![image](https://github.com/HIRU-VIRU/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145972122/027aab37-9706-48c9-b5ed-32b92b60ba12)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
