"""
models.py (Person 2) -- all models behind ONE interface.

Every model object must have:
    prepare(pool_texts)        # unsupervised setup on the whole pool (TF-IDF vocab, tokenizer, GloVe matrix...)
    fit(texts, y)              # train on raw texts + int labels
    predict(texts) -> (n,)     # class ids
    predict_proba(texts) -> (n, n_classes) numpy array, columns = classes 0..K-1

Person 2: add a "textcnn" entry (PyTorch + GloVe) to get_model(). Keep the same 4 methods.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class _TfidfBase:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        self.n_classes = None

    def prepare(self, pool_texts, n_classes=None):
        self.vec.fit(pool_texts)
        self.n_classes = n_classes

    def _full_proba(self, P, classes):
        # make sure output has a column for every class even if some were unseen in training
        if self.n_classes is None or P.shape[1] == self.n_classes:
            return P
        out = np.zeros((P.shape[0], self.n_classes))
        out[:, classes] = P
        return out


class TfidfLR(_TfidfBase):
    def __init__(self):
        super().__init__()
        self.clf = LogisticRegression(max_iter=300, C=5.0)

    def fit(self, texts, y):
        self.clf.fit(self.vec.transform(texts), y)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))

    def predict_proba(self, texts):
        return self._full_proba(self.clf.predict_proba(self.vec.transform(texts)), self.clf.classes_)


class TfidfSVM(_TfidfBase):
    """LinearSVC; probabilities = softmax of decision scores (good enough for uncertainty sampling)."""

    def __init__(self):
        super().__init__()
        self.clf = LinearSVC(C=1.0)

    def fit(self, texts, y):
        self.clf.fit(self.vec.transform(texts), y)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))

    def predict_proba(self, texts):
        d = self.clf.decision_function(self.vec.transform(texts))
        if d.ndim == 1:                      # binary case
            d = np.vstack([-d, d]).T
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return self._full_proba(e / e.sum(axis=1, keepdims=True), self.clf.classes_)


def get_model(name):
    if name == "tfidf_lr":
        return TfidfLR()
    if name == "tfidf_svm":
        return TfidfSVM()
    # Person 2: add  if name == "textcnn": return TextCNN()
    raise ValueError(f"unknown model {name}")
