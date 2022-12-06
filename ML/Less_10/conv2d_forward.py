import numpy as np


class Conv2d:

    def __init__(
            self, in_channels, out_channels, kernel_size_h, kernel_size_w, padding=0, stride=1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.padding = padding
        self.stride = stride

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self, ):
        np.random.seed(1)
        W = np.random.uniform(size=(
            self.in_channels, self.kernel_size_h, self.kernel_size_w, self.out_channels
        ))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def add_padding(self, x):
        if self.padding != 0:
            x_after_padding = np.zeros((
                x.shape[0], x.shape[1] + 2 * self.padding, x.shape[2] + 2 * self.padding
            ))

            h = x.shape[1]
            w = x.shape[2] + 2 * self.padding

            for i in range(x.shape[0]):
                padding_add_w = np.zeros((h, self.padding))
                padding_add_h = np.zeros((self.padding, w))

                x_after_padding[i] = np.concatenate(
                    (padding_add_h,
                     np.concatenate((padding_add_w, x[i], padding_add_w), axis=1),
                     padding_add_h), axis=0
                )

            return x_after_padding
        else:
            return x

    def forward(self, x):
        """
        [x] = in_channels x h x w
        W: in_channels x kernel_size_h x kernel_size_w x out_channels
        out: out_channels x out_fltr_h x out_fltr_w
        """
        self.W = self.W.transpose((3, 0, 1, 2))

        out_fltr_h = (x.shape[1] - self.kernel_size_h + 2 * self.padding + self.stride) // self.stride
        out_fltr_w = (x.shape[2] - self.kernel_size_w + 2 * self.padding + self.stride) // self.stride

        x = self.add_padding(x)

        output = np.zeros((self.out_channels, out_fltr_h, out_fltr_w,))

        for out_ch in range(self.W.shape[0]):
            for in_ch in range(self.W.shape[1]):
                for i in range(out_fltr_h):
                    for j in range(out_fltr_w):
                        output[out_ch, i, j] += np.sum(np.multiply(
                            x[in_ch, i * self.stride:i * self.stride + self.kernel_size_h,
                            j * self.stride:j * self.stride + self.kernel_size_w],
                            self.W[out_ch, in_ch, :, :]
                        ))

            output[out_ch, :, :] += self.biases[:, out_ch]

        return output


def read_matrix(in_channels, h, w, dtype=float):
    return np.array([list(map(dtype, input().split()))
                     for _ in range(in_channels * h)]).reshape(in_channels, h, w)


def print_matrix(matrix):
    for channel in matrix:
        for row in channel:
            print(' '.join(map(str, row)))


def solution():
    in_channels, out_channels, kernel_size_h, kernel_size_w, h, w, padding, stride = map(int, input().split())
    input_image = read_matrix(in_channels, h, w)

    conv = Conv2d(in_channels, out_channels, kernel_size_h, kernel_size_w, padding, stride)
    output = conv.forward(input_image).round(3)
    print_matrix(output)


if __name__ == '__main__':
    solution()
