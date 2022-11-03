import numpy as np


def get_accuracy(y_predict, y_true):
    return np.mean(np.equal(y_predict, y_true))


class LogisticRegression:

    def __init__(self, max_iter=1e3, lr=0.03, tol=0.001, penalty='l2', reg_coef=1e-4, print_every=100):

        """
        max_iter – максимальное количество
        """

        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.print_every = print_every

        self.weights = None
        self.bias = None
        self.penalty = penalty
        self.reg_coef = reg_coef

    def fit(self, X_train, y_train, X_val, y_val):

        """
        Обучение модели.

        X_train – матрица объектов для обучения
        y_train – ответы на объектах для обучения

        """

        n, m = X_train.shape

        self.weights = np.zeros((m, 1))
        self.bias = np.mean(y_train)

        n_iter = 0
        gradient_norm = np.inf

        while n_iter < self.max_iter and gradient_norm > self.tol:
            dJdw, dJdb = self.grads(X_train, y_train)
            gradient_norm = np.linalg.norm(np.hstack([dJdw.flatten(), [dJdb]]))

            self.weights = self.weights - self.lr * dJdw
            self.bias = self.bias - self.lr * dJdb

            n_iter += 1

            if n_iter % self.print_every == 0:
                self.print_metrics(X_train, y_train, X_val, y_val, n_iter, gradient_norm)

        return self

    def predict(self, X):

        """
        Метод возвращает предсказанную метку класса на объектах X
        """
        ans = self.predict_proba(X) > 0.5
        return ans.astype(int)

    def predict_proba(self, X):

        """
        Метод возвращает вероятность класса 1 на объектах X
        """
        return self.sigmoid(X.dot(self.weights) + self.bias)

    def grads(self, X, y):

        """
        Рассчёт градиентов
        """
        y_hat = self.predict_proba(X)

        if self.penalty == 'l2':
            dJdw = np.mean(X * (y_hat - y), axis=0, keepdims=True).T + self.reg_coef * self.weights
        elif self.penalty == 'l1':
            dJdw = np.mean(X * (y_hat - y), axis=0, keepdims=True).T + self.reg_coef * np.sign(self.weights)
        else:
            dJdw = np.mean(X * (y_hat - y), axis=0, keepdims=True).T
        dJdb = np.mean(y_hat - y)

        return dJdw, dJdb

    def print_metrics(self, X_train, y_train, X_val, y_val, n_iter, gradient_norm):

        train_predict = self.predict(X_train)
        val_predict = self.predict(X_val)

        accuracy_train = get_accuracy(train_predict, y_train)
        accuracy_val = get_accuracy(val_predict, y_val)

        print(f'{n_iter} completed. accuracy on train: {accuracy_train}, \
            val: {accuracy_val},  grad norm: {gradient_norm}')

    @staticmethod
    def sigmoid(x):
        """
        Сигмоида от x
        """
        return 1 / (1 + np.exp(-x))
