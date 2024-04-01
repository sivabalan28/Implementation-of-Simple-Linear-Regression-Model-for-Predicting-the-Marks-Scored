# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.    
2.Set variables for assigning dataset values.    
3.Import linear regression from sklearn.    
4.Assign the points for representing in the graph.    
5.Predict the regression for marks by using the representation of the graph.     
6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SIVAVBALAN S
RegisterNumber:  212222040100
```
```py
import pandas as pd
df=pd.read_csv('/content/TABLE - Sheet1.csv')
df.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/TABLE - Sheet1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.xlabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.xlabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
```

## Output:
## 1)HEAD:
![Screenshot 2024-04-01 135456](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/a31cc930-e688-46a9-9d23-1d1667a689d9)
## 2)GRAPH OF PLOTTED DATA:
![Screenshot 2024-04-01 135551](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/faaaaf6d-be7e-4798-b721-257fefe9cf8f)
## 3)TRAINED DATA:
![Screenshot 2024-04-01 135639](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/c81b6539-f686-4c48-b2ca-8b5d78d565e0)
## 4)LINE OF REGRESSION:
![Screenshot 2024-04-01 135816](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/2e4dd601-70a7-45c0-8ed7-8974394f01e0)
## 5)COEFFICIENT AND INTERCEPT VALUES:
![Screenshot 2024-04-01 135905](https://github.com/karthick960/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215938/35ec35e0-a3bc-4859-b3ae-af639b8556ca)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
