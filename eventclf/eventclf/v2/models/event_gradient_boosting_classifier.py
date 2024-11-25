import gensim.downloader
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer

from eventclf.v1.preprocessing import rich_analyzer_textual
from eventclf.v1.w2v_model import MaxEmbeddingVectorizer


class EventGradientBoostingClassifier:
    def __init__(self):
        self._vect_bow = CountVectorizer(tokenizer=rich_analyzer_textual)
        w2v_vectors = gensim.downloader.load("word2vec-google-news-300")
        self._vect_w2v = MaxEmbeddingVectorizer(w2v_vectors)
        self._model = GradientBoostingClassifier(
            random_state=4, n_estimators=100, max_depth=3
        )

    def _vectorise(self, text: pd.Series) -> np.array:
        bow = self._vect_bow.transform(text)
        w2v = self._vect_w2v.transform(text)
        return np.concatenate((bow.todense(), w2v), axis=1)

    def _tranform_input(self, df: pd.DataFrame) -> np.array:
        word_vectors = self._vectorise(df[self._text_column])
        input = np.concatenate(
            (word_vectors, df.drop(columns=self._text_column).to_numpy()), axis=1
        )
        return input

    def fit(self, x: pd.DataFrame, y: pd.Series, text_column: str = "tweet.text"):
        self._text_column = text_column
        self._vect_bow.fit(x[self._text_column])
        x_transformed = self._tranform_input(x)
        self._model.fit(x_transformed, y)

    def predict(self, x: pd.DataFrame) -> np.array:
        x_transformed = self._tranform_input(x)
        return self._model.predict(x_transformed)
