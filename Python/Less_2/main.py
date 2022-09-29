def counter(sentence):
    """Analog Counter() from collections."""
    dict_sentence = {}
    for word in sentence.lower().split():
        if word in dict_sentence:
            dict_sentence[word] += 1
        else:
            dict_sentence[word] = 1
    return dict_sentence


class CountVectorizer:
    def __init__(self, corpus=None):
        if corpus is None:
            corpus = []
        self.corpus = corpus

    def fit_transform(self, corpus):
        """"Returns a term-document matrix for a given corpus"""
        self.corpus = corpus
        seen = self.get_feature_names()
        term_doc_matrix = []
        for sentence in corpus:
            sentence = sentence.lower()
            dict_sentence = counter(sentence)
            counter_list = [dict_sentence[word] if word in sentence.split() else 0 for word in seen]
            term_doc_matrix.append(counter_list)
        return term_doc_matrix

    @staticmethod
    def get_feature_names():
        """Returns a list of features (unique words from the corpus)"""
        seen = []
        for i in range(len(corpus)):
            for word in corpus[i].split():
                if word not in seen:
                    seen.append(word.lower())
        return seen


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste"
    ]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(count_matrix)
