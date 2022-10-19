import numpy as np


def HimFunction(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def HimFunctionGradX(x, y):
    return 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)


def HimFunctionGradY(x, y):
    return 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)


def get_function_parameter_space(wrange=(-10, 10), brange=(-10, 10),
                                 num=100, function=HimFunction):
    """Function that generates matrixes of values for parameters"""

    w_grid = np.linspace(wrange[0], wrange[1], num=num)
    b_grid = np.linspace(brange[0], brange[1], num=num)

    W, B = np.meshgrid(w_grid, b_grid)
    J = function(W, B)

    return W, B, J


def find_minima(x, y, x_range=(-5, 5), y_range=(-5, 5), max_iteration=100,
                alpha=0.01, tol=1e-4, beta=0.5, v_x=0, v_y=0):
    """Function to Find Local Minimum with Gradient Descent and Momentum"""
    X, Y, J = get_function_parameter_space(wrange=x_range, brange=y_range,
                                           function=HimFunction)

    dJdx, dJdy = np.inf, np.inf
    iteration_number = 0
    while iteration_number < max_iteration and np.linalg.norm([dJdx, dJdy]) > tol:
        dJdx = np.clip(HimFunctionGradX(x, y), -5, 5)
        dJdy = np.clip(HimFunctionGradY(x, y), -5, 5)

        v_x = v_x * beta + dJdx * (1 - beta)
        x = x - alpha * v_x

        v_y = v_y * beta + dJdy * (1 - beta)
        y = y - alpha * v_y

        iteration_number += 1

    return x, y


def solution():
    x_0, y_0 = map(float, input().split())
    minima_coordinates = find_minima(x_0, y_0)
    result = ' '.join(map(str, minima_coordinates))
    print(result)


if __name__ == '__main__':
    solution()
