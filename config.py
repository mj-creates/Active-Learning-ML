"""config.py -- shared settings. EVERYONE uses these values so results are comparable."""

DATASET_NAMES = ["ag_news", "trec", "imdb"]
STRATEGIES = ["random", "least_confidence", "margin", "entropy", "qbc"]  # Person 3 adds e.g. "kmeans", "coreset"
SEEDS = [0, 1, 2, 3, 4]

INIT_SIZE = 50        # initial labelled samples
QUERY_SIZE = 50       # samples labelled per round
ROUNDS = 20           # total labelled = INIT_SIZE + ROUNDS * QUERY_SIZE

POOL_CAP = 8000       # pool subsample (speed)
TEST_CAP = 3000
QBC_MEMBERS = 5

# Result files MUST use exactly these columns:
RESULT_COLUMNS = ["dataset", "model", "strategy", "seed", "n_labelled", "acc", "macro_f1"]
