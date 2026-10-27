"""data.py (Person 1) -- loads AG News, TREC, IMDB. Returns RAW texts + numpy int labels."""
import numpy as np
from datasets import load_dataset
from config import POOL_CAP, TEST_CAP


def _load_trec():
    # TREC is an old script-based dataset, so load its parquet conversion instead
    attempts = [
        lambda: load_dataset("CogComp/trec", revision="refs/convert/parquet"),
        lambda: load_dataset("parquet", data_files={
            "train": "hf://datasets/CogComp/trec@refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "test": "hf://datasets/CogComp/trec@refs%2Fconvert%2Fparquet/default/test/0000.parquet"}),
    ]
    last = None
    for a in attempts:
        try:
            return a()
        except Exception as e:
            last = e
    raise RuntimeError(f"All TREC loading methods failed: {last}")


# name -> (loader, text column, label column)
DATASETS = {
    "ag_news": (lambda: load_dataset("fancyzhx/ag_news"), "text", "label"),
    "trec": (_load_trec, "text", "coarse_label"),
    "imdb": (lambda: load_dataset("stanfordnlp/imdb"), "text", "label"),
}


def load_data(name, pool_cap=POOL_CAP, test_cap=TEST_CAP, seed=123):
    """Returns pool_texts, pool_labels, test_texts, test_labels."""
    loader, tcol, lcol = DATASETS[name]
    ds = loader()
    tr, te = ds["train"], ds["test"]
    rng = np.random.RandomState(seed)
    ip = rng.choice(len(tr), min(pool_cap, len(tr)), replace=False)
    it = rng.choice(len(te), min(test_cap, len(te)), replace=False)
    return ([tr[int(i)][tcol] for i in ip], np.array([tr[int(i)][lcol] for i in ip]),
            [te[int(i)][tcol] for i in it], np.array([te[int(i)][lcol] for i in it]))


if __name__ == "__main__":
    for n in DATASETS:
        xt, yt, xs, ys = load_data(n)
        print(n, len(xt), len(xs), len(set(yt)))
