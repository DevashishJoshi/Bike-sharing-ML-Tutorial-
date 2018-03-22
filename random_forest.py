'''
cd VJTI Lab/Sem 6 Lab/ML/Bike-Sharing-Dataset
py -3 random_forest.py
'''
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr

train = pd.read_csv('day.csv')
train = train.sample(frac=1, random_state=0, axis=0)

ytrain_ = train['cnt']
Xtrain_ = train.drop(['cnt', 'registered', 'casual', 'instant', 'dteday'], axis=1)
m = Xtrain_.shape[0]
split_point = 600

Xtrain = Xtrain_[:split_point]
Xtest = Xtrain_[split_point:]
ytrain = ytrain_[:split_point]
ytest = ytrain_[split_point:]

mtrain = Xtrain.shape[0]
mtest = Xtest.shape[0]
n = Xtrain.shape[1]

r = rfr(random_state=0, n_estimators=9, max_depth=7, min_samples_split=4)
r.fit(Xtrain, ytrain)
important_features = pd.Series(data=r.feature_importances_,index=Xtrain.columns)
important_features.sort_values(ascending=False,inplace=True)
print(important_features)

train_acc = r.score(Xtrain, ytrain)*100
test_acc = r.score(Xtest, ytest)*100

print("Train accuracy : %.2f" % train_acc)
print("Test accuracy : %.2f" % test_acc)