"""al_loop.py (Person 1) -- the active learning loop. Few labels at the start, the rest unlabelled."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from config import INIT_SIZE, QUERY_SIZE, ROUNDS
from models import get_model
from strategies import select


def run_al(data, dataset, model_name, strategy, seed, verbose=True):
    pool_texts, y, test_texts, y_test = data
    n_classes = len(np.unique(y))
    rng = np.random.RandomState(seed)

    lab = np.array(rng.choice(len(y), INIT_SIZE, replace=False))
    unl = np.setdiff1d(np.arange(len(y)), lab)
    rows = []
    for r in range(ROUNDS + 1):
        model = get_model(model_name)
        model.prepare(pool_texts, n_classes)
        model.fit([pool_texts[i] for i in lab], y[lab])
        pred = model.predict(test_texts)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        rows.append(dict(dataset=dataset, model=model_name, strategy=strategy, seed=seed,
                         n_labelled=len(lab), acc=acc, macro_f1=f1))
        if verbose:
            print(f"[{dataset}|{model_name}|{strategy}|seed{seed}] round {r:2d} | "
                  f"labelled={len(lab):4d} | unlabelled={len(unl):5d} | acc={acc:.3f}")
        if r == ROUNDS:
            break
        pick = select(strategy, model, model_name, pool_texts, y, lab, unl, QUERY_SIZE, rng, n_classes)
        lab = np.concatenate([lab, pick])
        unl = np.setdiff1d(unl, pick)
    return rows
