# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. add a column to x for the intercept,initialize the theta
2. perform graadient descent
3. read the csv file
4. assuming the last column is ur target variable 'y' and the preceeding column
5. learn model parameters
6. predict target value for a new data point

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: A.ARUVI
RegisterNumber:  212222230014
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    #add a column to  x for the interccept
    X=np.c_[np.ones(len(X1)),X1]
    
    #initialize trhe Theeta
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    #perform gradient descent
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        
        #uptade theta using gradient descent
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

#assuming the last column is ur target variable 'y' and the preceeding column
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

#learn model parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

        

*/
```


## Output:

### X&Y values:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/fd6bb9aa-425a-4819-bed6-51c6ba29a0ce)
![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/223b16f2-ee3d-49b2-9912-e3e19e404f1f)


### X-Scaled & Y-Scaled:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/5dad4c50-9f7b-4f9a-b46f-ba5ebeaccdd1)

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/abec740a-5f42-4a67-8f45-69201b7f221a)

### Predicted value:

![image](https://github.com/Anandanaruvi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120443233/03525478-0253-435c-b9c2-63c0a885c3de)

## Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.


