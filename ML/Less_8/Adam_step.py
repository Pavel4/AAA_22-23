import numpy as np


class Adam:
    def __init__(self, num_features, lr=0.01, betas=(0.9, 0.99), eps=1e-6):
        shape = num_features

        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.v = np.zeros(shape)
        self.g = np.zeros(shape)
        self.v_hat = np.zeros(shape)
        self.g_hat = np.zeros(shape)

    def step(self, w, dLdw, t):
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * dLdw
        self.g = self.beta_2 * self.g + (1 - self.beta_2) * dLdw ** 2

        self.v_hat = self.v / (1 - self.beta_1 ** (t + 1))
        self.g_hat = self.g / (1 - self.beta_2 ** (t + 1))

        return w - self.lr / np.sqrt(self.g_hat + self.eps) * self.v_hat


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)]).astype(float)


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    iterations, num_features = map(int, input().split())
    lr, beta_1, beta_2, eps = map(float, input().split())
    w = read_matrix(1)
    dLdw = read_matrix(iterations)

    optimizer = Adam(num_features, lr, (beta_1, beta_2), eps)

    for i in range(iterations):
        w = optimizer.step(w, dLdw[i], i)

    print_matrix(w.round(3))


solution()
