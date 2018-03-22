'''
cd VJTI Lab/Sem 6 Lab/ML/Bike-Sharing-Dataset
py -3 gradient_descent.py
'''
from sklearn.metrics import r2_score
from prereq import gradDesc
#from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np, matplotlib.pyplot as p

#Read the data from csv files
df = pd.read_csv('day.csv')
df = df.sample(frac=1, random_state=0)

#Create X and y dataframes
y = df['cnt']
X = df.drop(['cnt', 'registered', 'casual', 'instant', 'dteday', 'mnth', 'weekday'], axis=1)
train_size = 600

#Mean normalization
temp_columns = ['temp', 'atemp', 'hum', 'windspeed']
X[temp_columns] = (X[temp_columns]-X[temp_columns].mean() )/X[temp_columns].std()

#One hot encoding of categorial features
categorial_features = ['season', 'weathersit']
X = pd.get_dummies(data=X, columns=categorial_features)
m, n = X.shape

#Split data into train and test sets
Xtrain = X[:train_size]
Xtest = X[train_size:]
ytrain = y[:train_size]
ytest = y[train_size:]
#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=3, train_size=train_size, test_size=m-train_size)

#Initialize theta for gradient descent
theta = pd.DataFrame([[0]*n]*1)
theta.columns = Xtrain.columns

#Perform gradient descent
dict = gradDesc(theta=theta, X=Xtrain, y=ytrain, alpha=0.04, num_iters=800)
theta = dict['theta'].transpose()
j_history = dict['j_history']

#Sort theta values
theta.columns = ['Importance']
theta = theta.sort_values('Importance', ascending=False)
print(theta)

#Plot graph of accuracy and cost vs number of iterations
p.plot(j_history)
p.xlabel('Number of iterations')
p.show()

#Predict on train and test set and find accuracy of the model
htrain = Xtrain.dot(theta)
htest = Xtest.dot(theta)
train_acc = r2_score(ytrain, htrain)*100
test_acc = r2_score(ytest, htest)*100

print("Train accuracy : %.2f" % train_acc)
print("Test accuracy : %.2f" % test_acc)