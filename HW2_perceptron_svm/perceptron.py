
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
plt.rcParams['figure.figsize'] = (10.0, 10.0)

class perceptron:
    def __init__(self, n_features: int, std: float):

        self.n_features = n_features
        self.weights = np.random.normal(0, std, self.n_features)

    def loss(self, X: np.ndarray, y: np.ndarray, reg_coeff: float):


        N = len(X)
        loss = np.sum((-1 * np.matmul(X, self.weights) * y)[np.where((-1 * np.matmul(X, self.weights) * y) > 0)]) / N
        loss += ((reg_coeff / 2) * np.dot(self.weights, self.weights))

        return loss

    def update_weights(self, X: np.ndarray, y: np.ndarray, learning_rate: float, reg_coeff: float):

        N = len(X)
        gradient = ((-1 * np.sum(((X.T * y).T)[np.where((-1 * np.matmul(X, self.weights) * y) > 0)], axis=0)) / N) + reg_coeff * self.weights
        self.weights -= learning_rate * gradient

    def predict(self, X):
        y_pred = []
        N = len(X)
        for i in range(N):
            if np.dot(X[i], self.weights) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)

        y_pred = np.array(y_pred)
        return y_pred


cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer["data"], cancer["target"]], columns = np.append(cancer["feature_names"],["target"]))

X = cancer["data"]
Y = cancer["target"]
Y[Y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

X_val = np.insert(X_val, 0, 1, axis=1)
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)

std = 0.0001
num_iters = 15000
reg_coeff = 0
learning_rate = 1e-9


model = perceptron(n_features=X_train.shape[1], std= std )
loss_history = []
loss_val_history = []
for it in range(num_iters):
    loss = model.loss(X_train, y_train, reg_coeff)
    loss_val = model.loss(X_val, y_val, reg_coeff)
    if it % 1000 == 0:
        val_preds =  model.predict(X_val)
        print('iteration %d, loss %f, val acc %.2f%%' % (it, loss,  accuracy_score(y_val,val_preds) * 100))
    model.update_weights(X_train, y_train, learning_rate , reg_coeff)
    loss_history.append(loss)
    loss_val_history.append(loss_val)



