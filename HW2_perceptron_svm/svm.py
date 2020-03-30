# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
plt.rcParams['figure.figsize'] = (10.0, 10.0)


class SVM:
    def __init__(self, n_features: int, std: float):
        """
        n_features: number of features in (or the dimension of) each instance
        std: standard deviation used in the initialization of the weights of svm
        """
        self.n_features = n_features
        ################################################################################
        # TODO: Initialize the weights of svm using random normal distribution with    #
        # standard deviation equals to std.                                            #
        ################################################################################

        # write your code here
        self.weights = np.random.normal(0, std, self.n_features)

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def loss(self, X: np.ndarray, y: np.ndarray, reg_coeff: float):
        """
        X: training instances as a 2d-array with shape (num_train, n_features)
        y: labels corresponsing to the given training instances as a 1d-array with shape (num_train,)
        reg_coeff: L2-regularization coefficient
        """
        loss = 0.0


        #################################################################################
        # TODO: Compute the hinge loss specified in the notebook and save it in the loss#                                                   # loss variable.                                                               #
        # NOTE: YOU ARE NOT ALLOWED TO USE FOR LOOPS!                                   #
        # Don't forget L2-regularization term in your implementation!                   #
        #################################################################################
        # write your code here
        N = len(X)
        loss = np.sum((1 - np.matmul(X, self.weights) * y)[np.where((1 - np.matmul(X, self.weights) * y) > 0)]) / N
        loss += ((reg_coeff / 2) * np.dot(self.weights, self.weights))


        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return loss

    def update_weights(self, X: np.ndarray, y: np.ndarray, learning_rate: float, reg_coeff: float):
        """
        Updates the weights of the svm using the gradient of computed loss with respect to the weights.
        learning_rate: learning rate that will be used in gradient descent to update the weights
        """
        ################################################################################
        # TODO: Compute the gradient of loss computed above w.r.t the svm weights.     #
        # and then update self.w with the computed gradient.                           #
        # (don't forget learning rate and reg_coeff in update rule)                    #
        # Don't forget L2-regularization term in your implementation!                  #
        ################################################################################

        # write your code here
        N = len(X)
        gradient = ((-1 * np.sum(((X.T * y).T)[np.where((1 - np.matmul(X, self.weights) * y) > 0)], axis=0)) / N) + reg_coeff * self.weights
        self.weights -= learning_rate * gradient


        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def predict(self, X):
        """
        X: Numpy 2d-array of instances
        """
        y_pred = None
        ################################################################################
        # TODO: predict the labels for the instances in X and save them in y_pred.     #                                      #
        ################################################################################

        # write your code here
        y_pred = []
        N = len(X)
        for i in range(N):
            if np.dot(X[i], self.weights) >= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)

        y_pred = np.array(y_pred)



        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
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
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

"""std = 0.0001
num_iters = 15000
reg_coeff = 20
learning_rate = 1e-8


model = SVM(n_features=X_train.shape[1], std= std)
loss_history = []
loss_val_history = []
for it in range(num_iters):
    loss = model.loss(X_train, y_train, reg_coeff)
    loss_val = model.loss(X_val, y_val, reg_coeff)
    if it % 100 == 0:
        val_preds =  model.predict(X_val)
        print('iteration %d, loss %f, val acc %f%%' % (it, loss,  accuracy_score(y_val, val_preds) * 100))
    model.update_weights(X_train, y_train, learning_rate , reg_coeff)
    loss_history.append(loss)
    loss_val_history.append(loss_val)
a = 0"""

std = 0.0001
batch_size = 200
num_iters = 15000
reg_coeff = 20
learning_rate=1e-8
model = SVM(n_features=X_train.shape[1], std= std )

loss_history = []
loss_val_history = []
for it in range(num_iters):
    X_batch = None
    y_batch = None
    ################################################################################
    # TODO: Sample batch_size elements from the training data and their            #
    # corresponding labels to use in this round of gradient descent.               #
    # Store the data in X_batch and their corresponding labels in                  #
    # y_batch; after sampling X_batch should have shape (batch_size, n_features)   #
    # and y_batch should have shape (batch_size,)                                  #
    #                                                                              #
    # Hint: Use np.random.choice to generate indices. Sampling with                #
    # replacement is faster than sampling without replacement.                     #
    ################################################################################

    #write your code here
    index = np.random.choice([i for i in range(len(X_train))], batch_size)
    X_batch = X_train[index]
    y_batch = y_train[index]

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    loss = model.loss(X_batch, y_batch, reg_coeff)
    loss_val = model.loss(X_val, y_val, reg_coeff)
    if it % 100 == 0:
        val_preds =  model.predict(X_val)
        print('iteration %d, loss %f, val acc %.2f%%' % (it, loss,  accuracy_score(y_val,val_preds) * 100))
    model.update_weights(X_batch, y_batch, learning_rate , reg_coeff)
    loss_history.append(loss)
    loss_val_history.append(loss_val)

