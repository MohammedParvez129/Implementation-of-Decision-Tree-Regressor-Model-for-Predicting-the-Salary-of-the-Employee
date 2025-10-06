# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import and Load Data:
Import libraries (pandas, sklearn) and load the employee salary dataset.

2.Preprocess Data:
Handle missing values, encode categorical data if needed, and define features (X) and target (y).

3.Split Data:
Split the dataset into training and testing sets using train_test_split().

4.Train Model:
Initialize a DecisionTreeRegressor and train it using the training data.

5.Predict and Evaluate:
Predict salary on the test set and evaluate using metrics like Mean Squared Error (MSE) or R² score.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOHAMMED PARVEZ S
RegisterNumber:  212223040113
*/
```

```python
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
```
<img width="278" height="127" alt="image" src="https://github.com/user-attachments/assets/ee173e4c-b82a-413a-8770-3eeddc887a7f" />

```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()
```
<img width="286" height="137" alt="image" src="https://github.com/user-attachments/assets/901e374c-3a33-45f7-a965-aba2aeb3ce02" />

```python
x=df[["Position","Level"]]
x.head()
```
<img width="214" height="132" alt="image" src="https://github.com/user-attachments/assets/70d3cd28-92eb-459d-bb0a-3938326bdf49" />

```python
y=df["Salary"]
y.head()
```
<img width="246" height="82" alt="image" src="https://github.com/user-attachments/assets/026c8ec0-39f5-4fd3-952e-d364ecf47787" />

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Name: MOHAMMED PARVEZ S")
print("RegNo: 212223040113")
print(y_pred)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("Mean Absolute Error:",mae)
print("R2 score:",r2)
dt.predict(pd.DataFrame([[5,6]],columns=["Position","Level"]))
```
## Output:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/13aea5bb-b6ed-43f5-8271-fbc662099375" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
