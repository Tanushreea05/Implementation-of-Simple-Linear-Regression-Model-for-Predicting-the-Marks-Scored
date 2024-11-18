# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, numpy, matplotlib, and scikit-learn.
2. Load the dataset `student_scores.csv` into a DataFrame and print it to verify contents.
3. Display the first and last few rows of the DataFrame to inspect the data structure.
4. Extract the independent variable (`x`) and dependent variable (`y`) as arrays from the DataFrame.
5. Split the data into training and testing sets, with one-third used for testing and a fixed `random_state` for reproducibility.
6. Create and train a linear regression model using the training data.
7. Make predictions on the test data and print both the predicted and actual values for comparison.
8. Plot the training data as a scatter plot and overlay the fitted regression line to visualize the model's fit.
9. Plot the test data as a scatter plot with the regression line to show model performance on unseen data.
10. Calculate and print error metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for evaluating model accuracy.
11. Display the plots to visually assess the regression results.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Tanushree.A
RegisterNumber:  212223100057
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
df.head()

![dfhead](https://user-images.githubusercontent.com/119393424/229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5.png)

df.tail()

![dftail](https://user-images.githubusercontent.com/119393424/229978854-6af7d9e9-537f-4820-a10b-ab537f3d0683.png)


Array value of X

![xvalue](https://user-images.githubusercontent.com/119393424/229978918-707c006d-0a30-4833-bf77-edd37e8849bb.png)

Array value of Y

![yvalue](https://user-images.githubusercontent.com/119393424/229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20.png)

Values of Y prediction

![ypred](https://user-images.githubusercontent.com/119393424/229979053-f32194cb-7ed4-4326-8a39-fe8186079b63.png)

Array values of Y test

![ytest](https://user-images.githubusercontent.com/119393424/229979114-3667c4b7-7610-4175-9532-5538b83957ac.png)

Training Set Graph

![train](https://user-images.githubusercontent.com/119393424/229979169-ad4db5b6-e238-4d80-ae5b-405638820d35.png)

Test Set Graph

![test](https://user-images.githubusercontent.com/119393424/229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1.png)


Values of MSE, MAE and RMSE

![mse](https://user-images.githubusercontent.com/119393424/229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c.png)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
