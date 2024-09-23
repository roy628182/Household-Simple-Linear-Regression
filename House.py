# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:45:32 2024

@author: roy62
"""


#Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#Importing DataSet 
house = pd.read_csv(r"E:\Data Science & AI\Dataset files\House_data.csv")
space=house['sqft_living']
price=house['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)


#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)

#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'green')
plt.plot(xtrain, regressor.predict(xtrain), color = 'black')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
