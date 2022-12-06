import numpy as np
import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        shape = (1, num_features)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if not torch.is_grad_enabled():

            return self.gamma * (x - self.moving_mean) / torch.sqrt(self.moving_var + self.eps) + self.beta
        else:
            mu = torch.mean(x, dim=0)
            var = torch.var(x, unbiased=False, dim=0)

            self.moving_mean = (1 - self.momentum) * self.moving_mean + self.momentum * torch.mean(x, dim=0)
            self.moving_var = (1 - self.momentum) * self.moving_var + self.momentum * torch.var(x, unbiased=True, dim=0)

            return self.gamma * (x - mu) / (torch.sqrt(var + self.eps)) + self.beta


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)]).astype(float)


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    batch_size, num_features = map(int, input().split())
    eps, momentum = map(float, input().split())
    train_vectors = read_matrix(batch_size)
    test_vectors = read_matrix(batch_size)

    train_vectors = torch.from_numpy(train_vectors).float()
    test_vectors = torch.from_numpy(test_vectors).float()

    batch_norm_1d = BatchNorm1d(num_features, eps, momentum)
    output_train = batch_norm_1d.forward(train_vectors).detach().numpy().round(2)
    with torch.no_grad():
        output_eval = batch_norm_1d.forward(test_vectors).detach().numpy().round(2)

    print_matrix(output_train)
    print()
    print_matrix(output_eval)


solution()
