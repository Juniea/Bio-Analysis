import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tianyi_blocked import BlockedPreprocessor, sel_folder, src_folder, stages, standard, seed

warnings.filterwarnings("ignore")

# config the script parameters
chunk_s     = 20.0      # contiguous chunk length (s) that is assigned to a fold as a whole
purge_s     = 2.0       # training rows closer than this to a test chunk are dropped
n_folds     = 5


def grouped_cv(token: str, skew: bool) -> dict:
    """
    Tianyi's frame-level model with a leakage-free CV: 20 s chunks are randomly assigned to 5 folds and training
    rows within 2 s of the test chunks are purged; SMOTE and z-score are fitted on the training rows only
    """
    with open(sel_folder / Path(f"{token}_P.json"), "r") as fp:
        cfg = json.load(fp)
    cfg.update(beh_path=src_folder / Path(f"{token}_B.csv"), sig_path=src_folder / Path(f"{token}_S.csv"),
               tvt_ratio=(1.0, 0.0, 0.0), shuffle=False, smote=None, normalizer=None)
    vote = pd.read_csv(sel_folder / Path(f"{token}_V.csv")).set_index("features")
    features = vote.loc[vote.sum(axis=1) >= standard].index.to_list()

    prep = BlockedPreprocessor(**cfg, log_level="WARNING", skew=skew)
    data = prep.get_origin_total(shard="all")         # aligned, interpolated to 0.1 s, time order, no transform
    x = data.loc[:, features].to_numpy()
    y = data.iloc[:, -1].to_numpy().astype(int)
    n = len(y)

    rows_per_chunk = int(round(chunk_s / 0.1))
    chunk = np.arange(n) // rows_per_chunk
    rng = np.random.default_rng(seed)
    fold_of_chunk = rng.permutation(np.arange(chunk.max() + 1) % n_folds)
    fold = fold_of_chunk[chunk]
    purge = int(round(purge_s / 0.1))

    proba = np.full(n, np.nan)
    for k in range(n_folds):
        test = fold == k
        near = np.convolve(test.astype(int), np.ones(2 * purge + 1, dtype=int), mode="same") > 0
        train = ~near
        if y[train].sum() < 6 or y[test].sum() == 0:
            continue
        x_tr, y_tr = SMOTE(sampling_strategy="auto", random_state=seed).fit_resample(x[train], y[train])
        scaler = StandardScaler().fit(x_tr)
        model = lgb.LGBMClassifier(
            boosting_type='gbdt', n_estimators=100, learning_rate=0.1, max_depth=10, num_leaves=31,
            verbose=-1, n_jobs=-1, random_state=seed,
        ).fit(scaler.transform(x_tr), y_tr)
        proba[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]

    ok = ~np.isnan(proba)
    yy, ss = y[ok], proba[ok]
    pp = (ss >= 0.5).astype(int)
    rf, nf = f1_score(yy, pp, zero_division=0), f1_score(yy, pp, pos_label=0, zero_division=0)
    return dict(token=token, skew=skew, rows=int(ok.sum()), reach_rows=int(yy.sum()), macro_f1=(rf + nf) / 2,
                reach_f1=rf, nonreach_f1=nf, bal_acc=balanced_accuracy_score(yy, pp), auc=roc_auc_score(yy, ss))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    tokens = sorted(
        p.stem[:-2] for p in sel_folder.glob("*_P.json")
        if p.stem.split("_")[2] in stages and (sel_folder / f"{p.stem[:-2]}_V.csv").exists()
    )
    rows = []
    for token in tokens:
        for skew in (False, True):
            r = grouped_cv(token, skew)
            rows.append(r)
            print(f"  {token} skew={skew!s:5s} macro={r['macro_f1']:.3f} reachF1={r['reach_f1']:.3f} "
                  f"bal_acc={r['bal_acc']:.3f} AUC={r['auc']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    if out:
        res.to_csv(out, index=False)
    print(res.groupby("skew")[["macro_f1", "reach_f1", "nonreach_f1", "bal_acc", "auc"]].mean().round(3).to_string())
