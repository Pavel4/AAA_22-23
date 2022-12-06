import numpy as np


class Conv1d:

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self, ):
        np.random.seed(1)
        W = np.random.uniform(size=(self.in_channels, self.kernel_size, self.out_channels))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        """
        [x] = in_channels x T
        w: in_channels x kernel_size x out_channels
        out: out_channels x T
        """

        self.W = self.W.transpose((2, 0, 1))

        if self.padding == 'same':
            padding_add = np.zeros((x.shape[0], self.kernel_size // 2))
            x = np.concatenate((padding_add, x, padding_add), axis=1)

        count_conv = x.shape[1] - self.kernel_size + 1
        output = np.zeros((self.out_channels, count_conv))

        for i in range(self.out_channels):
            for j in range(count_conv):
                output[i, j] = np.sum(np.multiply(x[:, j:j + self.kernel_size], self.W[i]))

        output += self.biases.T
        output = self.relu(output)
        
        return output

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    in_channels, out_channels, kernel_size = map(int, input().split())
    input_vectors = read_matrix(in_channels)

    conv = Conv1d(in_channels, out_channels, kernel_size)
    output = conv.forward(input_vectors).round(3)
    print_matrix(output)


if __name__ == '__main__':
    solution()
