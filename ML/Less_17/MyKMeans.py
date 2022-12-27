import numpy as np


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)  # np.sum((p1 - p2) ** 2)


def k_plus_plus(X: np.ndarray, k: int, random_state: int = 27) -> np.ndarray:
    """Инициализация центроидов алгоритмом k-means++.

    :param random_state: фиксируем
    :param X: исходная выборка
    :param k: количество кластеров
    :return: набор центроидов в одном np.array
    """
    np.random.seed = random_state
    centers = np.zeros((k, X.shape[1]))
    centers[0, :] = X[np.random.choice(len(X), 1), :]
    for c in range(1, k):
        ds = np.array([np.min([np.sum((dot - center) ** 2) for center in centers[:c]]) for dot in X])
        ps = ds / np.sum(ds)
        idx = np.random.choice(len(X), 1, p=ps)
        centers[c, :] = X[idx, :]
    return centers


class KMeans:
    def __init__(self, n_clusters=8, tol=0.0001, max_iter=300, random_state=None):
        self.n_iter_ = None
        self.inertia_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        labels = []

        # инициализируем центры кластеров
        centers = k_plus_plus(X, self.n_clusters)

        for n_iter in range(self.max_iter):
            # считаем расстояние от точек из X до центроидов
            distances = np.array([[distance(x, c) for c in centers] for x in X])

            # определяем метки как индекс ближайшего для каждой точки центроида
            labels = np.array([np.argmin(x_c_dist) for x_c_dist in distances])

            old_centers = centers.copy()
            for c in range(self.n_clusters):
                # пересчитываем центроид
                # новый центроид есть среднее точек X с меткой рассматриваемого центроида
                centers[c, :] = np.mean(X[labels == c, :], axis=0)

            # записываем условие сходимости
            # норма Фробениуса разности центров кластеров двух последовательных итераций < tol
            if np.linalg.norm((old_centers - centers), 'fro') < self.tol:
                break

        # cчитаем инерцию
        # сумма квадратов расстояний от точек до их ближайших центров кластеров
        inertia = np.sum([distance(X[i], centers[labels[i]]) ** 2 for i in range(len(labels))])

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    def predict(self, X):
        # определяем метку для каждого элемента X на основании обученных центров кластеров
        distances = np.array([[distance(x, c) for c in self.cluster_centers_] for x in X])
        labels = [np.argmin(x_c_dist) for x_c_dist in distances]
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_


def read_input():
    n1, n2, k = map(int, input().split())

    read_line = lambda x: list(map(float, x.split()))
    X_train = np.array([read_line(input()) for _ in range(n1)])
    X_test = np.array([read_line(input()) for _ in range(n2)])

    return X_train, X_test, k


def solution():
    X_train, X_test, k = read_input()
    kmeans = KMeans(n_clusters=k, tol=1e-8, random_state=27)
    kmeans.fit(X_train)
    train_labels = kmeans.labels_
    test_labels = kmeans.predict(X_test)

    print(' '.join(map(str, train_labels)))
    print(' '.join(map(str, test_labels)))


if __name__ == '__main__':
    solution()
