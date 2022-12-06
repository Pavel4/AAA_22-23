import numpy as np


class TfidfVectorizer:

    def __init__(self):
        self.count_docs = 0
        self.sorted_vocab = {}

    def fit(self, X):
        self.count_docs = len(X)

        for doc in X:
            for word in set(doc.split()):
                if word in self.sorted_vocab.keys():
                    self.sorted_vocab[word] += 1
                else:
                    self.sorted_vocab[word] = 1

        idf = [np.log(self.count_docs / value) for value in self.sorted_vocab.values()]
        self.sorted_vocab = dict(zip(self.sorted_vocab.keys(), idf))
        self.sorted_vocab = dict(sorted(self.sorted_vocab.items(), key=lambda item: item[0], reverse=False))

        return self

    def transform(self, X):
        count_docs_loc = len(X)
        ans = np.zeros((count_docs_loc, len(self.sorted_vocab)))
        for i in range(count_docs_loc):
            X_arr = X[i].split()
            len_X = len(X_arr)
            for word in X[i].split():
                if word in self.sorted_vocab.keys():
                    j = list(self.sorted_vocab.keys()).index(word)
                    ans[i, j] = X_arr.count(word) / len_X * self.sorted_vocab[word]

        return ans


def read_input():
    n1, n2 = map(int, input().split())

    train_texts = [input().strip() for _ in range(n1)]
    test_texts = [input().strip() for _ in range(n2)]

    return train_texts, test_texts


def solution():
    train_texts, test_texts = read_input()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    transformed = vectorizer.transform(test_texts)

    for row in transformed:
        row_str = ' '.join(map(str, np.round(row, 3)))
        print(row_str)


if __name__ == '__main__':
    solution()
