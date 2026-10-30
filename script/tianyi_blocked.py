import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src_predict.preprocessor import Preprocessor

warnings.filterwarnings("ignore")

# config the script parameters
beh         = "cont"
standard    = 3
seed        = 42
stages      = ("D1", "D2", "D7", "D8", "M4", "M5")
root        = Path(__file__).resolve().parents[1]
sel_folder  = root / Path(f"cache/rfecv_init_vote/{beh}")
src_folder  = root / Path("data/initialization")


class BlockedPreprocessor(Preprocessor):
    """
    Tianyi's Preprocessor with two optional changes:
    - reverse: reverse the time order before the (unshuffled) 50/50 split, so train = second half, test = first half
    - skew: permute every cell's RAW (~1 Hz) signal on the time axis before the akima interpolation
    """

    def __init__(self, *args, reverse: bool = False, skew: bool = False, **kwargs):
        self.reverse = reverse
        self.skew = skew
        super().__init__(*args, **kwargs)

    def _config_signal(self, deci: int = 1) -> None:
        if self.skew:
            rng = np.random.default_rng(seed)
            frame = self.signal_df.pop("x0000")
            self.signal_df = self.signal_df.apply(lambda c: rng.permutation(c.values))
            self.signal_df.insert(0, "x0000", frame)
        super()._config_signal(deci=deci)

    def _transform(self, *args, **kwargs) -> None:
        if self.reverse:
            self.dataset_df = self.dataset_df.iloc[::-1]
        super()._transform(*args, **kwargs)


def evaluate(token: str, reverse: bool, skew: bool) -> dict:
    with open(sel_folder / Path(f"{token}_P.json"), "r") as fp:
        cfg = json.load(fp)
    cfg["beh_path"] = src_folder / Path(f"{token}_B.csv")
    cfg["sig_path"] = src_folder / Path(f"{token}_S.csv")
    cfg["shuffle"] = False   # blocked: first 50 % / last 50 % of the session

    vote = pd.read_csv(sel_folder / Path(f"{token}_V.csv")).set_index("features")
    features = vote.loc[vote.sum(axis=1) >= standard].index.to_list()

    prep = BlockedPreprocessor(**cfg, log_level="WARNING", reverse=reverse, skew=skew)
    train_x = prep.get_train(shard="x").loc[:, features]
    train_y = prep.get_train(shard="y").values
    test_x = prep.get_test(shard="x").loc[:, features]
    test_y = prep.get_test(shard="y").values.astype(int)

    # identical model to src_predict.rfecv_predict
    model = lgb.LGBMClassifier(
        boosting_type='gbdt', n_estimators=100, learning_rate=0.1, max_depth=10, num_leaves=31,
        verbose=-1, n_jobs=-1, random_state=seed,
    ).fit(train_x, train_y)
    p = model.predict(test_x).astype(int)
    s = model.predict_proba(test_x)[:, 1]

    two = 0 < test_y.sum() < len(test_y)
    reach_f1 = f1_score(test_y, p, zero_division=0) if two else np.nan
    nonreach_f1 = f1_score(test_y, p, pos_label=0, zero_division=0)
    return dict(
        token=token, half="second" if not reverse else "first", skew=skew, n=len(test_y), reach=int(test_y.sum()),
        macro_f1=(reach_f1 + nonreach_f1) / 2 if two else np.nan, reach_f1=reach_f1, nonreach_f1=nonreach_f1,
        bal_acc=balanced_accuracy_score(test_y, p) if two else np.nan, auc=roc_auc_score(test_y, s) if two else np.nan,
    )


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    tokens = sorted(
        p.stem[:-2] for p in sel_folder.glob("*_P.json")
        if p.stem.split("_")[2] in stages and (sel_folder / f"{p.stem[:-2]}_V.csv").exists()
    )
    rows = []
    for token in tokens:
        for skew in (False, True):
            for reverse in (False, True):
                try:
                    r = evaluate(token, reverse, skew)
                except ValueError as e:     # the training half has a single class (no reach frames), SMOTE fails
                    print(f"  {token} test={'first' if reverse else 'second'} skew={skew}: skipped ({e})", flush=True)
                    continue
                rows.append(r)
                print(f"  {token} test={r['half']:6s} skew={skew!s:5s} reach={r['reach']:4d} "
                      f"macro={r['macro_f1']:.3f} reachF1={r['reach_f1']:.3f} AUC={r['auc']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    if out:
        res.to_csv(out, index=False)
    cols = ["macro_f1", "reach_f1", "nonreach_f1", "bal_acc", "auc"]
    print(res.groupby("skew")[cols].mean().round(3).to_string())
