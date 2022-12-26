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

        return preds

    def get_best_split(self, X, y):

        best_Q = -1
        best_j = 0
        best_t = 0
        best_left_ids = []
        best_right_ids = []

        n, m = X.shape[0], X.shape[1]

        for j in range(m):
            X_set = np.unique(X[:, j])
            thresholds = [(X_set[i] + X_set[i + 1]) / 2 for i in range(len(X_set) - 1)]
            for threshold in thresholds:
                left_ids = [X[:, j] <= threshold]
                right_ids = [X[:, j] > threshold]

                left_tree = y[tuple(left_ids)]
                right_tree = y[tuple(right_ids)]
                Q = self.calc_Q(y, left_tree, right_tree)
                if Q > best_Q:
                    best_Q = Q
                    best_j = j
                    best_t = threshold
                    best_left_ids = left_ids
                    best_right_ids = right_ids

        return best_j, best_t, best_left_ids, best_right_ids

    def calc_Q(self, y, y_left, y_right):
        if self.criterion == 'mse':
            return self.mse(y) - (y_left.shape[0] / y.shape[0] * self.mse(y_left)
                                  + y_right.shape[0] / y.shape[0] * self.mse(y_right))
        else:
            return self.mae(y) - (y_left.shape[0] / y.shape[0] * self.mae(y_left)
                                  + y_right.shape[0] / y.shape[0] * self.mae(y_right))

    def mse(self, y):
        n = y.shape[0]
        y_mean = np.mean(y)
        return np.sum([(y_ - y_mean) ** 2 for y_ in y]) / n

    def mae(self, y):
        n = y.shape[0]
        y_median = np.median(y)
        return np.sum([abs(y_ - y_median) for y_ in y]) / n

    def _build_tree(self, curr_node, X, y):
        if curr_node['depth'] == self.max_depth:  # выход из рекурсии если построили до максимальной глубины
            if self.criterion == 'mse':
                curr_node['value'] = np.mean(y)
            else:
                curr_node['value'] = np.median(y)
            return

        if len(np.unique(y)) == 1:  # выход из рекурсии значения если "y" одинковы для все объектов
            if self.criterion == 'mse':
                curr_node['value'] = np.mean(y)
            else:
                curr_node['value'] = np.median(y)
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
