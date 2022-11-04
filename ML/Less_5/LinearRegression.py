import numpy as np
import numbers


def get_mape(y_predict, y_true):
    return (abs(y_predict - y_true) / y_true).mean()


class LinearRegression:

    def __init__(self, max_iter=1e4, lr=0.001, tol=0.001, print_every=100):

        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.print_every = print_every

        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train, X_val, y_val):

        self.check_regression_X_y(X_train, y_train)
        self.check_regression_X_y(X_val, y_val)

        n, m = X_train.shape

        self.weights = np.zeros((m, 1))
        self.bias = np.median(y_train)

        n_iter = 0
        gradient_norm = np.inf

        # Add pulse
        v_x, v_y = 0, 0
        beta = 0.1

        while n_iter < self.max_iter and gradient_norm > self.tol:

            dJdw, dJdb = self.grads(X_train, y_train)

            gradient_norm = np.linalg.norm(np.hstack([dJdw.flatten(), [dJdb]]))

            v_x = v_x * beta + dJdw * (1 - beta)
            self.weights = self.weights - self.lr * v_x

            v_y = v_y * beta + dJdb * (1 - beta)
            self.bias = self.bias - self.lr * v_y

            n_iter += 1

            if n_iter % self.print_every == 0:
                self.print_metrics(X_train, y_train, X_val, y_val, n_iter, gradient_norm)

        return self

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias

    def grads(self, X, y):

        lam = 1e-3
        y_hat = self.predict(X)
        dJdw = 2 * X.T @ (y_hat - y) / X.shape[0]  # lam * np.linalg.norm(y_hat)
        dJdb = 2 * (y_hat - y).mean()

        self.check_grads(dJdw, dJdb)

        return dJdw, dJdb

    def print_metrics(self, X_train, y_train, X_val, y_val, n_iter, gradient_norm):

        train_preds = self.predict(X_train)
        val_preds = self.predict(X_val)

        MAPE_train = get_mape(train_preds, y_train)
        MAPE_val = get_mape(val_preds, y_val)

        print(f'{n_iter} completed. MAPE on train: {MAPE_train}, val: {MAPE_val},  grad norm: {gradient_norm}')

    def check_grads(self, dJdw, dJdb):

        if not isinstance(dJdb, numbers.Real):
            raise ValueError(f'Производная по параметру b должна быть действительным '
                             f'числом, как и сам параметр b, а у нас {dJdb} типа {type(dJdb)}')

        if dJdw.shape != self.weights.shape:
            raise ValueError(f'Размерность градиента по параметрам w должна совпадать с самим вектором w, '
                             f'а у нас dJdw.shape = {dJdw.shape} не совпадает с weight.shape = {self.weights.shape}')

    @staticmethod
    def check_regression_X_y(X, y):

        if X.shape[0] == 0:
            raise ValueError(f'X и y не должны быть пустыми, а у нас X.shape = {X.shape} и y.shape = {y.shape}')

        if np.isnan(X).any():
            raise ValueError(f'X не должен содержать "not a number" (np.nan)')

        if np.isnan(y).any():
            raise ValueError(f'y не должен содержать "not a number" (np.nan)')

        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Длина X и y должна быть одинаковой, а у нас X.shape = {X.shape}, y.shape = {y.shape}')

        if y.shape[1] != 1:
            raise ValueError(f'y - вектор ответов должен быть размерности (m, 1), а у нас y.shape = {y.shape}')

        if np.any([(not isinstance(value, numbers.Real)) for value in y.flatten()]):
            raise ValueError(f'Ответы на объектах должны быть действительными числами!')
