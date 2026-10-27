"""
run_experiments.py -- run ONE model on all datasets/strategies/seeds and save ONE csv.

Usage (Colab or terminal):
    python run_experiments.py --model tfidf_lr
    python run_experiments.py --model tfidf_svm --datasets trec imdb
Output: results_<model>.csv   (columns: dataset, model, strategy, seed, n_labelled, acc, macro_f1)
Upload that csv to the GitHub repo. Person 4 merges all results_*.csv.
"""
import argparse
import pandas as pd
import config
from data import load_data
from al_loop import run_al

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--datasets", nargs="+", default=config.DATASET_NAMES)
ap.add_argument("--strategies", nargs="+", default=config.STRATEGIES)
ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
args = ap.parse_args()

all_rows = []
for ds in args.datasets:
    data = load_data(ds)
    for strat in args.strategies:
        for seed in args.seeds:
            all_rows += run_al(data, ds, args.model, strat, seed, verbose=(seed == args.seeds[0]))
    pd.DataFrame(all_rows)[config.RESULT_COLUMNS].to_csv(f"results_{args.model}.csv", index=False)  # save as we go

print(f"saved results_{args.model}.csv")
