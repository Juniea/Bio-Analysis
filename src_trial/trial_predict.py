import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, ConfusionMatrixDisplay

from tools import SyncLogger, check_folders, fetch_tokens


BEH_FPS = 7.06

# the positive behaviors of each prediction task (all other kept events are negatives)
TASKS = {
    "cont&inv": {"cont reach", "cont reaching", "cont in vain"},
    "cont": {"cont reach", "cont reaching"},
}

# events that are not mouse behaviors, therefore never used as samples
# ("noise generation" is the rig's 5 s white-noise marker ~5.7 s after valve opened in the trials without a reach,
#  not a behavior; all spellings found in the B csv files are listed)
EXCLUDED = {
    "valve opened",
    "two photon imaging",
    "two-photon imaging",
    "last 2p",
    "last 2p frame",
    "last frame",
    "noise generation",
    "noise genearation",
    "noist generation",
}


def _normalize(name: str) -> str:
    """
    Normalize the behavior name, e.g. ' Cont  reach (1)' -> 'cont reach'
    """
    key = " ".join(str(name).strip().lower().split())
    if "(" in key:
        key = key[:key.index("(")].strip()
    return key


def build_events(
    beh_path: str | os.PathLike | Path,
    sig_path: str | os.PathLike | Path,
    left: float = 3.0,
    right: float = 4.0,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Cut one window per behavior event, centered at the event start, ranging [start - left, start + right] seconds

    :param beh_path: The path to the behavior file
    :param sig_path: The path to the signal file
    :param left: The seconds before the event start
    :param right: The seconds after the event start

    :return: (event table [trial, behavior, start], windows [n_events, n_cells, n_lags], feature names)
    """

    beh = pd.read_csv(beh_path)
    sig = pd.read_csv(sig_path)
    if "x0000" in sig.columns:
        sig = sig.set_index("x0000")

    # the same time conversions as src_predict.Preprocessor
    sig_fps = 1 / (1.000045 if Path(sig_path).stem.split('_')[0] == 'K215' else 1.001286)
    sig_time = sig.index.to_numpy(dtype=float) / sig_fps
    beh_start = (beh["Start"] + BEH_FPS * 0.5) / BEH_FPS

    # the first column is the trial number, only filled at the valve-opened row of each trial
    events = pd.DataFrame({
        "trial": beh.iloc[:, 0].ffill().fillna(0).astype(int),
        "behavior": beh["Behavior"].map(_normalize),
        "start": beh_start,
    })
    events = events.loc[events["start"].notna() & ~events["behavior"].isin(EXCLUDED)]

    # keep the events whose window is fully covered by the signal
    lags = np.arange(-left, right + 1)
    events = events.loc[(events["start"] - left >= sig_time[0]) & (events["start"] + right <= sig_time[-1])]
    events = events.reset_index(drop=True)

    # interpolate every cell at the 1-second spaced window timestamps
    values = sig.to_numpy(dtype=float)
    windows = np.stack([
        np.stack([np.interp(start + lags, sig_time, values[:, c]) for c in range(values.shape[1])])
        for start in events["start"]
    ]) if len(events) else np.empty((0, values.shape[1], len(lags)))

    names = [f"{cell}_t{int(lag):+d}" for cell in sig.columns for lag in lags]
    return events, windows, names


def f1_scores(real: np.ndarray, pred: np.ndarray) -> dict:
    """
    The macro F1, the positive (task behaviors) F1 and the negative (other behaviors) F1
    """
    neg, pos = f1_score(real, pred, labels=[0, 1], average=None, zero_division=0)
    return {"test_f1_macro": (neg + pos) / 2, "test_f1_pos": pos, "test_f1_neg": neg}


def _make_model(params: dict, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**params, subsample_freq=1, verbose=-1, n_jobs=1, random_state=seed)


def _tune(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_trials: int, seed: int) -> dict:
    """
    Optuna search maximizing the mean balanced accuracy of a trial-grouped stratified CV on the training set
    """

    n_splits = int(min(5, np.bincount(y).min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(x, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 25, 500, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 64, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        scores = []
        for train_idx, valid_idx in folds:
            model = _make_model(params, seed).fit(x[train_idx], y[train_idx])
            scores.append(balanced_accuracy_score(y[valid_idx], model.predict(x[valid_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "cv_balanced_accuracy": study.best_value}


def _predict_token(
    token: str,
    src_folder: Path,
    task: str,
    left: float,
    right: float,
    test_size: float,
    n_trials: int,
    min_class: int,
    seed: int,
) -> tuple[dict, pd.DataFrame | None]:
    events, windows, names = build_events(
        beh_path=src_folder / f"{token}_B.csv",
        sig_path=src_folder / f"{token}_S.csv",
        left=left,
        right=right,
    )
    y = events["behavior"].isin(TASKS[task]).to_numpy(dtype=int)
    record = {"token": token, "n_events": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    if min(record["n_pos"], record["n_neg"]) < min_class:
        record["status"] = "skipped: too few events in one class"
        return record, None

    # split by trial so that all events of one trial fall in the same side
    x = windows.reshape(len(windows), -1)
    groups = events["trial"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(x, y, groups))

    best = _tune(x[train_idx], y[train_idx], groups[train_idx], n_trials=n_trials, seed=seed)
    params = {k: v for k, v in best.items() if k != "cv_balanced_accuracy"}
    model = _make_model(params, seed).fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])

    majority = np.bincount(y[train_idx]).argmax()
    record.update({
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "test_pos": int(y[test_idx].sum()),
        "test_accuracy": accuracy_score(y[test_idx], pred),
        "test_balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
        **f1_scores(y[test_idx], pred),
        "majority_accuracy": accuracy_score(y[test_idx], np.full(len(test_idx), majority)),
        **best,
        "status": "ok",
    })

    raw = events.loc[test_idx, ["trial", "behavior", "start"]].assign(token=token, real=y[test_idx], prediction=pred)
    return record, raw


@check_folders("tgt_folder")
def trial_predict(
    src_folders: tuple[Path, ...],
    tgt_folder: str | os.PathLike | Path,
    task: str           = "cont&inv",
    left: float         = 3.0,
    right: float        = 4.0,
    test_size: float    = 0.2,
    n_trials: int       = 50,
    min_class: int      = 10,
    seed: int           = 42,
    n_jobs: int         = -1,
) -> pd.DataFrame:
    """
    Event-level prediction: one sample per behavior event (window centered at the event start),
    trial-grouped train / test split, LightGBM tuned by Optuna on the training set only

    :param src_folders: The folders that contain the B & S csv files
    :param tgt_folder: The folder where the summary, the raw predictions and the confusion matrix will be placed
    :param task: The key of TASKS that decides the positive behaviors
    :param left: The seconds before the event start
    :param right: The seconds after the event start
    :param test_size: The ratio of the testing set
    :param n_trials: The number of Optuna trials for each token
    :param min_class: The least number of events of each class for a token to be predicted
    :param seed: The random seed
    :param n_jobs: The number of tokens processed in parallel

    :return: The summary table
    """

    logger = SyncLogger(level="INFO")
    jobs = [(token, Path(folder)) for folder in src_folders for token in fetch_tokens(folder=Path(folder), shard="all")]
    logger.info(role="trial_predict", message=f"[{task}] {len(jobs)} tokens, {n_trials} optuna trials each")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_predict_token)(token, folder, task, left, right, test_size, n_trials, min_class, seed)
        for token, folder in jobs
    )

    summary = pd.DataFrame([record for record, _ in results])
    raw = pd.concat([r for _, r in results if r is not None], ignore_index=True)
    summary.to_csv(Path(tgt_folder) / "summary.csv", index=False)
    raw.to_csv(Path(tgt_folder) / "raw.csv", index=False)

    # pooled confusion matrix over the testing sets of all tokens
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        raw["real"],
        raw["prediction"],
        display_labels=["Other", task],
        cmap="Reds",
        im_kw={"vmin": 0.0, "vmax": 1.0},
        normalize="true",
        ax=ax,
    )
    plt.savefig(Path(tgt_folder) / "matrix.pdf")
    plt.close()

    logger.info(role="trial_predict", message=f"[{task}] summary & raw & matrix saved to {tgt_folder}")
    return summary
