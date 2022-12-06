import numpy as np


class RNN:

    def __init__(self, in_features, hidden_size, n_classes, activation='tanh'):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.activation = activation

    def init_weight_matrix(self, size):
        np.random.seed(1)
        W = np.random.uniform(size=size)
        return W

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    def forward(self, x):
        a = np.zeros((self.hidden_size, 1))
        T = x.shape[1]
        W_ax = self.init_weight_matrix((self.hidden_size, self.in_features))
        W_aa = self.init_weight_matrix((self.hidden_size, self.hidden_size))
        W_ya = self.init_weight_matrix((self.n_classes, self.hidden_size))
        b_a = self.init_weight_matrix((self.hidden_size,))
        b_y = self.init_weight_matrix((self.n_classes,))
        output = np.zeros((self.n_classes, T))

        for i in range(T):
            a = np.tanh((W_aa @ a).reshape((self.hidden_size,)) + W_ax @ x[:, i] + b_a)
            output[:, i] = self.softmax((W_ya @ a) + b_y)

        return output


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_features, hidden_size, n_classes = map(int, input().split())
    input_vectors = read_matrix(in_features)

    rnn = RNN(in_features, hidden_size, n_classes)
    output = rnn.forward(input_vectors).round(3)
    print_matrix(output)


if __name__ == '__main__':
    solution()
