'''
cd VJTI Lab/Sem 6 Lab/ML/Bike-Sharing-Dataset
py -3 gradient_descent_with_kfold_cross_validation.py
'''
from sklearn.metrics import r2_score
from prereq import cost, grad, gradDesc
from sklearn.model_selection import train_test_split, KFold
import pandas as pd, numpy as np, matplotlib.pyplot as p, random

#Read the data from csv files
df = pd.read_csv('day.csv')
df = df.sample(frac=1, random_state=0)

#Create X and y dataframes
y = df['cnt']
X = df.drop(['cnt', 'registered', 'casual', 'instant', 'dteday', 'mnth', 'weekday'], axis=1)

#Mean normalization
temp_columns = ['temp', 'atemp', 'hum', 'windspeed']
X[temp_columns] = (X[temp_columns]-X[temp_columns].mean() )/X[temp_columns].std()

#One hot encoding of categorial features
categorial_features = ['season', 'weathersit']
X = pd.get_dummies(data=X, columns=categorial_features)
m, n = X.shape

#Initialize variables to perform k-fold cross validation
kf = KFold(n_splits=4, shuffle=True, random_state=0)
train_acc = []
test_acc = []

#Perform 4-fold cross validation
for train_index, test_index in kf.split(X):
	Xtrain, Xtest = X.loc[train_index], X.loc[test_index]
	ytrain, ytest = y.loc[train_index], y.loc[test_index]
	
	#Initialize theta for gradient descent
	theta = pd.DataFrame([[0]*n]*1)
	theta.columns = Xtrain.columns

	#Perform gradient descent
	dict = gradDesc(theta=theta, X=Xtrain, y=ytrain, alpha=0.03, num_iters=800)
	theta = dict['theta'].transpose()
	j_history = dict['j_history']

	#Plot graph of accuracy and cost vs number of iterations
	p.plot(j_history)

	#Predict on train and test set and find accuracy of the model
	htrain = Xtrain.dot(theta)
	htest = Xtest.dot(theta)
	train_acc.append(float(r2_score(ytrain, htrain)*100))
	test_acc.append(float(r2_score(ytest, htest)*100))

print([int(i) for i in train_acc])
print([int(i) for i in test_acc])

p.xlabel('Number of iterations')
p.show()

train_acc = sum(train_acc) / float(len(train_acc))
test_acc = sum(test_acc) / float(len(test_acc))

print("Train accuracy : %.2f" % train_acc)
print("Test accuracy : %.2f" % test_acc)