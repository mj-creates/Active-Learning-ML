"""
strategies.py (Person 3) -- query strategies.

select(...) returns an array of k indices (into the pool) to label next. Always choose from `unl`.
Person 3: add "kmeans" / "coreset" (diversity) and optionally a hybrid. Keep the same function signature.
"""
import numpy as np
from config import QBC_MEMBERS
from models import get_model


def _uncertainty(P, strategy):
    if strategy == "least_confidence":
        return 1.0 - P.max(axis=1)
    if strategy == "margin":
        s = np.sort(P, axis=1)
        return -(s[:, -1] - s[:, -2])
    if strategy == "entropy":
        return -(P * np.log(P + 1e-12)).sum(axis=1)
    raise ValueError(strategy)


def _qbc(model_name, pool_texts, y, lab, unl, rng, n_classes):
    votes = np.zeros((len(unl), n_classes))
    for _ in range(QBC_MEMBERS):
        idx = rng.choice(lab, len(lab), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        m = get_model(model_name)
        m.prepare(pool_texts, n_classes)
        m.fit([pool_texts[i] for i in idx], y[idx])
        pred = m.predict([pool_texts[i] for i in unl])
        votes[np.arange(len(unl)), pred] += 1
    V = votes / QBC_MEMBERS
    return -(V * np.log(V + 1e-12)).sum(axis=1)   # vote entropy


def select(strategy, model, model_name, pool_texts, y, lab, unl, k, rng, n_classes):
    if strategy == "random":
        return rng.choice(unl, k, replace=False)
    if strategy == "qbc":
        scores = _qbc(model_name, pool_texts, y, lab, unl, rng, n_classes)
    elif strategy in ("least_confidence", "margin", "entropy"):
        scores = _uncertainty(model.predict_proba([pool_texts[i] for i in unl]), strategy)
    else:
        raise ValueError(f"unknown strategy {strategy}")   # Person 3: add kmeans / coreset above
    return unl[np.argsort(-scores)[:k]]
