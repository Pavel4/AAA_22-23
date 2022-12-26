import numpy as np


class MyDecisionTreeRegressor:

    def __init__(self, max_depth=None, max_features=None, min_leaf_samples=None, criterion='mse'):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_samples = min_leaf_samples
        self._node = {
            'left': None,
            'right': None,
            'feature': None,
            'threshold': None,
            'depth': 0,
            'value': None
        }
        self.tree = None  # словарь в котором будет храниться построенное дерево
        self.criterion = criterion

    def fit(self, X, y):

        self.tree = {'root': self._node.copy()}  # создаём первую узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self

    def predict(self, X):
        preds = []
        for x in X:
            preds_for_x = self._get_predict(self.tree['root'], x)
            preds.append(preds_for_x)

        return np.array(preds)

    def get_best_split(self, X, y):

        best_Q = -1
        best_j = 0
        best_t = 0
        best_left_ids = []
        best_right_ids = []

        n, m = X.shape[0], X.shape[1]
        y_mse = self.mse(y)
        y_shape = y.shape[0]
        for j in range(m):
            X_set = np.unique(X[:, j])
            thresholds = (X_set[1:] + X_set[:-1]) / 2
            for threshold in thresholds:
                left_ids = [X[:, j] <= threshold]
                right_ids = [X[:, j] > threshold]

                left_tree = y[tuple(left_ids)]
                right_tree = y[tuple(right_ids)]
                Q = self.calc_Q(y_mse, y_shape, left_tree, right_tree)
                if Q > best_Q:
                    best_Q = Q
                    best_j = j
                    best_t = threshold
                    best_left_ids = left_ids
                    best_right_ids = right_ids

        return best_j, best_t, best_left_ids, best_right_ids

    def calc_Q(self, y_mse, y_shape, y_left, y_right):

        return y_mse - (y_left.shape[0] / y_shape * self.mse(y_left)
                        + y_right.shape[0] / y_shape * self.mse(y_right))

    def mse(self, y):

        y_mean = np.mean(y)

        return np.square(y - y_mean).mean()

    def mae(self, y):
        n = y.shape[0]
        y_median = np.median(y)
        return np.sum([abs(y_ - y_median) for y_ in y]) / n

    def _build_tree(self, curr_node, X, y):
        if curr_node['depth'] == self.max_depth:  # выход из рекурсии если построили до максимальной глубины
            curr_node['value'] = np.mean(y)
            return

        if len(np.unique(y)) == 1:  # выход из рекурсии значения если "y" одинковы для все объектов
            curr_node['value'] = np.mean(y)
            return

        j, t, left_ids, right_ids = self.get_best_split(X, y)  # нахождение лучшего разбиения

        curr_node['feature'] = j  # признак по которому производится разбиение в текущем узле
        curr_node['threshold'] = t  # порог по которому производится разбиение в текущем узле

        left = self._node.copy()  # создаём узел для левого поддерева
        right = self._node.copy()  # создаём узел для правого поддерева

        left['depth'] = curr_node['depth'] + 1  # увеличиваем значение глубины в узлах поддеревьев
        right['depth'] = curr_node['depth'] + 1

        curr_node['left'] = left
        curr_node['right'] = right

        self._build_tree(left, X[left_ids], y[left_ids])  # продолжаем построение дерева
        self._build_tree(right, X[right_ids], y[right_ids])

    def _get_predict(self, node, x):
        if node['threshold'] is None:  # если в узле нет порога, значит это лист, выходим из рекурсии
            return node['value']

        if x[node['feature']] <= node[
            'threshold']:  # уходим в правое или левое поддерево в зависимости от порога и признака
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)


class MyGradientBoostingRegressor:

    def __init__(self, learning_rate=0.1, max_depth=None, max_features=None, n_estimators=100, init='mean', tol=1e-4):
        self.y_init = None
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.estimators = []
        self.init = init
        self.tol = tol

    def fit(self, X, y):

        if self.init == 'mean':
            self.y_init = np.mean(y)
        else:
            self.y_init = 0.0

        y_ans = np.full(y.shape, self.y_init)
        for _ in range(self.n_estimators):
            dtr = MyDecisionTreeRegressor(max_depth=self.max_depth,
                                          max_features=self.max_features,
                                          criterion='mse')
            dtr.fit(X=X, y=(y - y_ans))
            self.estimators.append(dtr)
            y_ans += self.learning_rate * dtr.predict(X).reshape(y_ans.shape)
            if abs(y - y_ans).any() < self.tol:
                break
        return self.estimators

    def predict(self, X):
        y_ans = np.full((1, X.shape[0]), self.y_init)
        for dtr in self.estimators:
            y_ans += self.learning_rate * dtr.predict(X)

        return y_ans


def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix


def read_input_matriсes(n, _, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train, X_test


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matriсes(n, m, k)

    gb = MyGradientBoostingRegressor()
    gb.fit(X_train, y_train)

    predictions = gb.predict(X_test)
    print_matrix(predictions)


if __name__ == '__main__':
    solution()
