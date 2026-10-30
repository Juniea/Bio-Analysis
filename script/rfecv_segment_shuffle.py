import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, recall_score, accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_predict.preprocessor import Preprocessor

# config the script parameters
token       = "K227_B_D2"
beh         = "cont"
standard    = 3
seed        = 42
n_perm      = 20
root        = Path(__file__).resolve().parents[1]
sel_folder  = root / Path(f"cache/rfecv_init_vote/{beh}")
src_folder  = root / Path("data/initialization")


def run_lengths(y: np.ndarray) -> list[tuple[int, int]]:
    """
    Split a 0/1 label array into consecutive runs

    :param y: The 0/1 label array

    :return: [(value, length), ...] in the original temporal order
    """
    change = np.flatnonzero(np.diff(y)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(y)]
    return [(int(y[s]), int(e - s)) for s, e in zip(starts, ends)]


def segment_shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle the order of the 1-runs among themselves and the 0-runs among themselves,
    keeping the 0/1 alternation, the starting value and every run length

    :param y: The 0/1 label array
    :param rng: The random generator

    :return: The surrogate label array with the same run-length statistics
    """
    runs = run_lengths(y)
    ones = [l for v, l in runs if v == 1]
    zeros = [l for v, l in runs if v == 0]
    rng.shuffle(ones)
    rng.shuffle(zeros)
    ones, zeros = iter(ones), iter(zeros)

    out = [np.full(next(ones) if v == 1 else next(zeros), v) for v, _ in runs]
    return np.concatenate(out).astype(y.dtype)


class SegmentShufflePreprocessor(Preprocessor):
    """
    Same pipeline as Preprocessor, except the aligned labels are replaced by a segment-shuffled surrogate
    BEFORE the row shuffle / train-test split / SMOTE / normalization
    """

    def __init__(self, *args, label_seed: int | None = None, **kwargs):
        self.label_seed = label_seed
        super().__init__(*args, **kwargs)

    def _align(self, deci: int = 1) -> None:
        super()._align(deci=deci)
        if self.label_seed is not None:
            y = self.dataset_df.loc[:, "Label"].values
            self.dataset_df.loc[:, "Label"] = segment_shuffle(y, np.random.default_rng(self.label_seed))


class PreInterpSkewPreprocessor(Preprocessor):
    """
    Same pipeline as Preprocessor, except every cell's RAW (~1 Hz) signal is permuted on the time axis
    BEFORE the akima interpolation to 0.1 s, so the signal has no relation to behavior at all
    """

    def __init__(self, *args, skew_seed: int = 42, **kwargs):
        self.skew_seed = skew_seed
        super().__init__(*args, **kwargs)

    def _config_signal(self, deci: int = 1) -> None:
        rng = np.random.default_rng(self.skew_seed)
        frame = self.signal_df.pop("x0000") if "x0000" in self.signal_df.columns else None
        self.signal_df = self.signal_df.apply(lambda c: rng.permutation(c.values))
        if frame is not None:
            self.signal_df.insert(0, "x0000", frame)
        super()._config_signal(deci=deci)


def evaluate(prep: Preprocessor, features: list[str], skew: bool = False) -> dict:
    train_x = prep.get_train(shard="x").loc[:, features]
    train_y = prep.get_train(shard="y").values
    test_x = prep.get_test(shard="x").loc[:, features]
    test_y = prep.get_test(shard="y").values

    # identical to the `skew` control in src_predict.rfecv_predict
    if skew:
        np.random.seed(seed)
        train_x = train_x.apply(np.random.permutation)

    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        num_leaves=31,
        verbose=-1,
        n_jobs=-1,
        random_state=seed
    )
    model.fit(train_x, train_y)
    test_p = model.predict(test_x)

    return {
        "auc": roc_auc_score(test_y, model.predict_proba(test_x)[:, 1]),
        "macro_f1": f1_score(test_y, test_p, average="macro"),
        "reach_f1": f1_score(test_y, test_p, pos_label=1),
        "nonreach_f1": f1_score(test_y, test_p, pos_label=0),
        "acc": accuracy_score(test_y, test_p),
        "bal_acc": balanced_accuracy_score(test_y, test_p),
        "recall_0": recall_score(test_y, test_p, pos_label=0),
        "recall_1": recall_score(test_y, test_p, pos_label=1),
    }


if __name__ == '__main__':
    with open(sel_folder / Path(f'{token}_P.json'), 'r') as fp:
        prep_config = json.load(fp)
    prep_config["beh_path"] = src_folder / Path(f"{token}_B.csv")
    prep_config["sig_path"] = src_folder / Path(f"{token}_S.csv")

    # --blocked: no row shuffle before the 50/50 split, i.e. first half of the session = train, second half = test
    if "--blocked" in sys.argv:
        prep_config["shuffle"] = False
        print("[blocked split] train = first 50 % of the session, test = last 50 %")

    vote = pd.read_csv(sel_folder / Path(f"{token}_V.csv")).set_index('features')
    features = vote.loc[vote.sum(axis=1) >= standard].index.to_list()

    # real labels: run-length summary
    prep = Preprocessor(**prep_config, log_level="WARNING")
    y = prep.get_origin_total(shard="y").values.astype(int)
    runs = run_lengths(y)
    dt = 0.1  # seconds per row after the 0.1 s re-gridding
    print(f"[{token}] rows={len(y)}  features={len(features)}  positive ratio={y.mean():.3f}")
    print(f"  1-runs (s, in order): {[round(l * dt, 1) for v, l in runs if v == 1]}")
    print(f"  0-runs (s, in order): {[round(l * dt, 1) for v, l in runs if v == 0]}")

    rows = [{"condition": "real labels", **evaluate(prep, features)}]
    rows.append({"condition": "real labels + skew (Tianyi)", **evaluate(prep, features, skew=True)})
    for k in range(5):
        pp = PreInterpSkewPreprocessor(**prep_config, log_level="WARNING", skew_seed=k)
        rows.append({"condition": f"skew BEFORE interpolation #{k}", **evaluate(pp, features)})

    df = pd.DataFrame(rows).set_index("condition")
    pd.set_option("display.width", 200)
    print(df.round(3))

    if "--segment" in sys.argv:
        perm = pd.DataFrame([
            evaluate(SegmentShufflePreprocessor(**prep_config, log_level="WARNING", label_seed=k), features)
            for k in range(n_perm)
        ])
        print(f"\nsegment shuffle over {n_perm} permutations (mean ± sd):")
        print((perm.mean().round(3).astype(str) + " ± " + perm.std().round(3).astype(str)).to_string())
