
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('RiderDataCSV.csv')

# Splitting Price Columns
# Auto Column
new = df["Auto"].str.split(" - ", n = 1, expand = True)
df["First"] = new[0].astype(int)
df["Second"] = new[1].astype(int)
df.drop(columns = ["Auto"], inplace = True)
df["Auto"] = df[["First", "Second"]].mean(axis = 1)

# Prime Column
new = df["Prime"].str.split(" - ", n = 1, expand = True)
df["First"] = new[0].astype(int)
df["Second"] = new[1].astype(int)
df.drop(columns = ["Prime"], inplace = True)
df["Prime"] = df[["First", "Second"]].mean(axis = 1)

# Mini Column
new = df["Mini"].str.split(" - ", n = 1, expand = True)
df["First"] = new[0].astype(int)
df["Second"] = new[1].astype(int)
df.drop(columns = ["Mini"], inplace = True)
df["Mini"] = df[["First", "Second"]].mean(axis = 1)

# Micro Column
new = df["Micro"].str.split(" - ", n = 1, expand = True)
df["First"] = new[0].astype(int)
df["Second"] = new[1].astype(int)
df.drop(columns = ["Micro"], inplace = True)
df["Micro"] = df[["First", "Second"]].mean(axis = 1)

df.drop(columns = ["First"], inplace = True)
df.drop(columns = ["Second"], inplace = True)

# Encoding categorical data
time = pd.get_dummies(df['Time'], drop_first = True)
df = pd.concat([df, time], axis = 1)
df.drop(['Time'], axis = 1, inplace = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = df[['Pickup Latitude', 'Pickup Longitude', 'Drop Latitude', 'Drop Longitude', 'Morning', 'Noon']]
y = df[['Auto', 'Prime', 'Mini', 'Micro']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)