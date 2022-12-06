import numpy as np


def decision_stump(X, y):
    best_Q = -1
    best_j = 0  # индекс признака по которому производился лучший сплит
    best_t = 0  # порог с котором сравнивается признак
    best_left_ids = []  # вектор со значениями True для объектов в левом поддереве, остальные False
    best_right_ids = []  # вектор со значениями True для объектов в правом поддереве, остальные False
    y_preds_left = 0  # предсказание в левом поддерева
    y_preds_right = 0  # предсказание в правом поддерева

    n, m = X.shape[0], X.shape[1]

    y_mean = y.mean()
    H_y = np.square(y - y_mean).mean()

    for j in range(m):
        X_set = np.unique(X[:, j])
        thresholds = [(X_set[i] + X_set[i + 1]) / 2 for i in range(len(X_set) - 1)]
        for threshold in thresholds:
            left_ids = [X[:, j] <= threshold]
            right_ids = [X[:, j] > threshold]

            left_tree = y[tuple(left_ids)]
            right_tree = y[tuple(right_ids)]

            H_l = np.square(left_tree - left_tree.mean()).mean()
            H_r = np.square(right_tree - right_tree.mean()).mean()

            Q = H_y - (left_tree.shape[0] / n * H_l + right_tree.shape[0] / n * H_r)

            if Q > best_Q:
                best_Q = Q
                best_j = j
                best_t = threshold
                best_left_ids = left_ids
                best_right_ids = right_ids
                y_preds_left = left_tree.sum() / left_tree.shape[0]
                y_preds_right = right_tree.sum() / right_tree.shape[0]

    result = [
        best_Q,
        best_j,
        best_t,
        np.array(best_left_ids).sum(),
        np.array(best_right_ids).sum(),
        y_preds_left,
        y_preds_right
    ]
    return result


def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train


def solution():
    X, y = read_input()
    result = decision_stump(X, y)
    result = np.round(result, 2)
    output = ' '.join(map(str, result))
    print(output)


if __name__ == '__main__':
    solution()
