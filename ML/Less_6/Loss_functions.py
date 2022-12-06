import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z_ = z.copy()
    for i in range(z_.shape[0]):
        z_[i, :] = np.exp(z_[i, :]) / np.sum(np.exp(z_[i, :]))

    return z_


def logloss(y, y_hat):
    sum_ = 0
    for i in range(y.shape[0]):
        for k in range(y.shape[1]):
            sum_ += y[i, k] * np.log(y_hat[i, k]) + (1 - y[i, k]) * np.log(1 - y_hat[i, k])

    return -sum_


def cross_entropy(y, y_hat):
    sum_ = 0
    for i in range(y.shape[0]):
        for k in range(y.shape[1]):
            sum_ += y[i, k] * np.log(y_hat[i, k])

    return -sum_


if __name__ == '__main__':
    y = np.array([[1, 0], [0, 1]])
    z = np.array([[-3.0, 2.0], [0.0, 1.0]])

    logloss_value = logloss(y, sigmoid(z))
    crossentropy_value = cross_entropy(y, softmax(z))

    logloss_value = str(np.round(logloss_value, 3))
    crossentropy_value = str(np.round(crossentropy_value, 3))
    print(logloss_value + ' ' + crossentropy_value)
