import numpy as np, pandas as pd
from sklearn.metrics import r2_score

def cost(theta, X, y):
	m = X.shape[0]
	h = X.dot(theta.transpose()).values.reshape(m,)
	J = 1.0/(2*m) * sum(np.square(h-y))
	return J

def grad(theta, X, y):
	m = y.shape[0]
	h = X.dot(theta.transpose()).values.reshape(m,)
	g = (1.0/m) * ( (X.transpose()).dot(h-y) )
	return g.transpose()

def gradDesc(theta, X, y, alpha, num_iters):
	m = X.shape[0]
	n = X.shape[1]
	j_history = np.zeros( (num_iters,2) )
	j_history = pd.DataFrame(j_history, columns=['Cost', 'Accuracy'])
	temp = alpha
	for i in range(0, num_iters):
		theta[:] = theta[:] - alpha*grad(theta, X, y)
		h = X.dot(theta.transpose()).values.reshape(m,)
		j_history.iloc[i, 0] = cost(theta, X, y)
		acc = r2_score(y, h)
		j_history.iloc[i, 1] = acc*100
	return {'theta':theta, 'j_history':j_history}