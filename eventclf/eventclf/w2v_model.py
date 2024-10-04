import gensim.downloader
import numpy as np

w2v_vectors = gensim.downloader.load('word2vec-google-news-300')

ndim = len(w2v_vectors['excited'])
print("number of the vector", ndim)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = ndim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class SumEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = ndim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.sum([self.word2vec[w] for w in words if w in self.word2vec]
                   or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class MaxEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = ndim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.max([self.word2vec[w] for w in words if w in self.word2vec]
                   or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class MixEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = ndim

    def fit(self, X, y):
        return self

    def transform(self, X):
        x_repr = [[self.word2vec[w] for w in words if w in self.word2vec]
                  or [np.zeros(self.dim)]
                  for words in X]
        return np.array([np.concatenate([
            np.sum(repr_, axis=0),
            np.max(repr_, axis=0)
        ], axis=0)
            for repr_ in x_repr
        ])