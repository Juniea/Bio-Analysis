import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from tools import SyncLogger, check_folders, fetch_tokens
from .trial_predict import TASKS, f1_scores
from .deepset_predict import build_spans, _split, _best_threshold, _metrics


CELL_FEATURES = ("mean", "max", "min", "std", "auc", "slope", "first_last", "peak_time", "pre_mean", "event_minus_pre")
CELL_AGGREGATES = ("mean", "std", "max", "min", "median", "q75", "q90")

# LightGBM tree / sampling / penalty settings
LGBM_PRESETS = {
    "default": {"num_leaves": 31, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "regularized": {"num_leaves": 15, "min_child_samples": 80, "subsample": 0.7, "colsample_bytree": 0.3, "reg_lambda": 10.0},
    "strong": {"num_leaves": 7, "max_depth": 3, "min_child_samples": 150, "subsample": 0.6, "colsample_bytree": 0.2,
               "reg_lambda": 30.0, "reg_alpha": 1.0, "min_split_gain": 0.01},
    "very_strong": {"num_leaves": 4, "max_depth": 2, "min_child_samples": 300, "subsample": 0.5, "colsample_bytree": 0.1,
                    "reg_lambda": 100.0, "reg_alpha": 5.0, "extra_trees": True},
}


def cell_statistics(window: np.ndarray, pre: float = 2.0, post: float = 2.0) -> np.ndarray:
    """
    The per-cell statistics of one event window (1 point per second)

    :param window: [n_cells, n_points]
    :param pre: The seconds before the event start inside the window
    :param post: The seconds after the event end inside the window

    :return: [n_cells, len(CELL_FEATURES)]
    """
    n_points = window.shape[1]
    t = np.arange(n_points, dtype=float) - (n_points - 1) / 2
    n_pre, n_post = int(round(pre)), int(round(post))
    event = window[:, n_pre:max(n_pre + 1, n_points - n_post)]
    pre_mean = window[:, :n_pre].mean(axis=1)
    return np.stack([
        window.mean(axis=1),
        window.max(axis=1),
        window.min(axis=1),
        window.std(axis=1),
        window.sum(axis=1),                                         # AUC with 1-second steps
        (window * t).sum(axis=1) / max((t ** 2).sum(), 1e-6),       # least-squares slope
        window[:, -1] - window[:, 0],
        window.argmax(axis=1) / max(n_points - 1, 1),               # relative peak position in [0, 1]
        pre_mean,
        event.mean(axis=1) - pre_mean,
    ], axis=1)


EXT_CELL_FEATURES = (
    # distribution
    "mean", "median", "max", "min", "std", "range", "q25", "q75", "q90", "skew", "kurtosis", "energy",
    # temporal
    "auc", "slope", "first_last", "peak_time", "onset_time", "mean_abs_diff", "max_abs_diff", "autocorr1", "n_peaks",
    # segments
    "pre_mean", "pre_max", "pre_std", "event_mean", "event_max", "event_std", "post_mean", "post_max",
    "event_minus_pre", "post_minus_event",
)
EXT_CELL_AGGREGATES = ("mean", "std", "max", "min", "median", "q10", "q25", "q75", "q90", "top5_mean")
EXT_POPULATION_FEATURES = (
    "pop_mean", "pop_max", "pop_std", "pop_slope", "pop_event_minus_pre",
    "synchrony", "responsive_frac_0.5", "responsive_frac_1", "active_frac",
)


def _segments(n_points: int, pre: float, post: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The column indices of the pre / event / post segments of a window
    """
    idx = np.arange(n_points)
    n_pre, n_post = int(round(pre)), int(round(post))
    return idx[:n_pre], idx[n_pre:max(n_pre + 1, n_points - n_post)], (idx[-n_post:] if n_post > 0 else idx[-1:])


def extended_cell_statistics(
    window: np.ndarray, pre: float = 2.0, post: float = 2.0, valid: np.ndarray | None = None,
) -> np.ndarray:
    """
    The extended per-cell statistics of one event window (1 point per second)

    :param window: [n_cells, n_points]
    :param valid: [n_points] boolean, the time points kept (the masked ones are ignored); None keeps all

    :return: [n_cells, len(EXT_CELL_FEATURES)]
    """
    n_cells, n_points = window.shape
    valid = np.ones(n_points, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.full((n_cells, len(EXT_CELL_FEATURES)), np.nan)
    kept = np.flatnonzero(valid)
    w = window[:, kept]
    t = kept - (n_points - 1) / 2
    t = t - t.mean()

    def segment(cols: np.ndarray) -> np.ndarray:
        cols = cols[valid[cols]]
        return window[:, cols] if len(cols) else np.full((n_cells, 1), np.nan)

    pre_cols, event_cols, post_cols = _segments(n_points, pre, post)
    pre_part, event, post_part = segment(pre_cols), segment(event_cols), segment(post_cols)

    mean = w.mean(axis=1)
    std = w.std(axis=1)
    centered = w - mean[:, None]
    full_centered = window - mean[:, None]
    safe_std = np.where(std > 1e-6, std, np.nan)
    # differences between consecutive kept points; lag-1 pairs & peaks only where all the points involved are kept
    diff = np.abs(np.diff(w, axis=1)) if w.shape[1] > 1 else np.zeros((n_cells, 1))
    pairs = valid[:-1] & valid[1:]
    lag1 = (full_centered[:, :-1] * full_centered[:, 1:])[:, pairs].mean(axis=1) / safe_std ** 2 \
        if pairs.any() else np.full(n_cells, np.nan)
    peaks = np.zeros(n_cells)
    if n_points > 2:
        middle = window[:, 1:-1]
        triples = valid[:-2] & valid[1:-1] & valid[2:]
        peaks = ((middle > window[:, :-2]) & (middle >= window[:, 2:]) & (middle > 1))[:, triples].sum(axis=1)
    above = w > 1
    onset = np.where(above.any(axis=1), kept[above.argmax(axis=1)] / max(n_points - 1, 1), 1.0)
    q25, q75, q90 = np.quantile(w, [0.25, 0.75, 0.90], axis=1)

    with np.errstate(all="ignore"):
        return np.stack([
            mean, np.median(w, axis=1), w.max(axis=1), w.min(axis=1), std,
            w.max(axis=1) - w.min(axis=1), q25, q75, q90,
            (centered ** 3).mean(axis=1) / safe_std ** 3,
            (centered ** 4).mean(axis=1) / safe_std ** 4,
            (w ** 2).mean(axis=1),
            mean * n_points,
            (w * t).sum(axis=1) / max((t ** 2).sum(), 1e-6),
            w[:, -1] - w[:, 0],
            kept[w.argmax(axis=1)] / max(n_points - 1, 1),
            onset,
            diff.mean(axis=1), diff.max(axis=1), lag1, peaks,
            pre_part.mean(axis=1), pre_part.max(axis=1), pre_part.std(axis=1),
            event.mean(axis=1), event.max(axis=1), event.std(axis=1),
            post_part.mean(axis=1), post_part.max(axis=1),
            event.mean(axis=1) - pre_part.mean(axis=1),
            post_part.mean(axis=1) - event.mean(axis=1),
        ], axis=1)


def extended_event_features(
    window: np.ndarray, pre: float = 2.0, post: float = 2.0, valid: np.ndarray | None = None,
) -> np.ndarray:
    """
    The extended per-cell statistics aggregated over the cells, plus the population-level features

    :param valid: [n_points] boolean, the time points kept (see extended_cell_statistics); None keeps all

    :return: [len(EXT_CELL_FEATURES) * len(EXT_CELL_AGGREGATES) + len(EXT_POPULATION_FEATURES)]
    """
    n_points = window.shape[1]
    valid = np.ones(n_points, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    stats = extended_cell_statistics(window, pre, post, valid)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        top = np.sort(np.nan_to_num(stats, nan=-np.inf), axis=0)[-5:]
        top = np.where(np.isinf(top), np.nan, top)
        aggregated = np.stack([
            np.nanmean(stats, axis=0),
            np.nanstd(stats, axis=0),
            np.nanmax(stats, axis=0),
            np.nanmin(stats, axis=0),
            np.nanmedian(stats, axis=0),
            *np.nanquantile(stats, [0.10, 0.25, 0.75, 0.90], axis=0),
            np.nanmean(top, axis=0),
        ], axis=1)

        # population trace & synchrony over the kept time points
        kept = np.flatnonzero(valid)
        w = window[:, kept]
        pop_full = window.mean(axis=0)
        pop = pop_full[kept]
        t = kept - (n_points - 1) / 2
        t = t - t.mean() if len(t) else t
        pre_cols, event_cols, _ = _segments(n_points, pre, post)
        pop_event, pop_pre = pop_full[event_cols[valid[event_cols]]], pop_full[pre_cols[valid[pre_cols]]]
        z = (w - w.mean(axis=1, keepdims=True)) / w.std(axis=1, keepdims=True)
        z = z[np.isfinite(z).all(axis=1)]
        if len(z) > 1 and w.shape[1] > 1:
            corr = z @ z.T / w.shape[1]
            synchrony = corr[np.triu_indices(len(z), k=1)].mean()
        else:
            synchrony = np.nan
        event_minus_pre = stats[:, EXT_CELL_FEATURES.index("event_minus_pre")]
        responsive = np.isfinite(event_minus_pre).any()
        population = np.array([
            pop.mean(), pop.max(), pop.std(), (pop * t).sum() / max((t ** 2).sum(), 1e-6),
            pop_event.mean() - pop_pre.mean(),
            synchrony,
            (event_minus_pre > 0.5).mean() if responsive else np.nan,
            (event_minus_pre > 1.0).mean() if responsive else np.nan,
            (w.max(axis=1) > w.min(axis=1)).mean(),
        ]) if len(kept) else np.full(len(EXT_POPULATION_FEATURES), np.nan)
    return np.concatenate([aggregated.reshape(-1), population])


def overlap_mask(
    events: pd.DataFrame,
    y: np.ndarray,
    rule: str,
    pre: float = 2.0,
    post: float = 2.0,
    fixed_after: float | None = None,
) -> np.ndarray:
    """
    Flag the events overlapping the positive events of the same token (see drop_overlap of lgbm_cellstats)

    :return: boolean [n_events]
    """
    starts, ends = events["start"].to_numpy(), events["end"].to_numpy()
    pos_starts, pos_ends = starts[y == 1], ends[y == 1]
    w0 = starts - pre
    w1 = ends + post if fixed_after is None else starts + fixed_after

    flags = np.zeros(len(events), dtype=bool)
    for i in range(len(events)):
        if rule == "event":
            flags[i] = ((np.minimum(ends[i], pos_ends) - np.maximum(starts[i], pos_starts)) > 0).any()
        else:
            inside = np.clip(np.minimum(w1[i], pos_ends) - np.maximum(w0[i], pos_starts), 0, None).sum()
            if rule == "window_any":
                flags[i] = inside > 0
            elif rule == "window25":
                flags[i] = inside >= 0.25 * (w1[i] - w0[i])
            else:
                raise ValueError(f"unknown drop_overlap: {rule}")
    return flags


def first_negatives(
    events: pd.DataFrame,
    y: np.ndarray,
    keep: np.ndarray,
    pre: float = 2.0,
    post: float = 2.0,
    fixed_after: float | None = None,
) -> np.ndarray:
    """
    Keep only the first of the overlapping non-reach windows: in time order, a kept non-reach is dropped when its window
    overlaps the window of the last non-reach still kept (the reaches & the already dropped events are left as they are)

    :param keep: boolean [n_events], the events kept by the previous cleaning
    :return: boolean [n_events], False for the dropped non-reaches
    """
    starts, ends = events["start"].to_numpy(), events["end"].to_numpy()
    w0 = starts - pre
    w1 = ends + post if fixed_after is None else starts + fixed_after

    flags = np.ones(len(events), dtype=bool)
    last_end = -np.inf
    for i in np.argsort(starts, kind="stable"):
        if y[i] == 1 or not keep[i]:
            continue
        if w0[i] < last_end:
            flags[i] = False
        else:
            last_end = w1[i]
    return flags


def context_mask(
    events: pd.DataFrame,
    pre: float = 2.0,
    fixed_after: float = 4.0,
) -> list[np.ndarray]:
    """
    For every fixed window [start - pre, start + fixed_after] (one point per second), flag the points kept: a point
    (covering +-0.5 s) is masked when it overlaps any other event of the token, unless it also overlaps the event itself

    :param events: All the events of one token (before any cleaning), with start / end in seconds

    :return: one boolean [n_points] array per event, True = kept
    """
    starts, ends = events["start"].to_numpy(), events["end"].to_numpy()
    offsets = np.arange(-pre, fixed_after + 1e-6, 1.0)
    masks = []
    for i in range(len(events)):
        lo, hi = starts[i] + offsets - 0.5, starts[i] + offsets + 0.5
        own = (lo < ends[i]) & (hi > starts[i]) | ((lo <= starts[i]) & (hi > starts[i]))
        overlap = (lo[:, None] < ends[None, :]) & (hi[:, None] > starts[None, :])
        overlap[:, i] = False
        masks.append(own | ~overlap.any(axis=1))
    return masks


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "basic":
        return [f"{f}_{a}" for f in CELL_FEATURES for a in CELL_AGGREGATES]
    return [f"{f}_{a}" for f in EXT_CELL_FEATURES for a in EXT_CELL_AGGREGATES] + list(EXT_POPULATION_FEATURES)


def event_features(window: np.ndarray, pre: float = 2.0, post: float = 2.0) -> np.ndarray:
    """
    Aggregate the per-cell statistics over the cells, so that any cell number gives the same feature length

    :return: [len(CELL_FEATURES) * len(CELL_AGGREGATES)], ordered feature-major
    """
    stats = cell_statistics(window, pre, post)
    aggregated = np.stack([
        stats.mean(axis=0),
        stats.std(axis=0),
        stats.max(axis=0),
        stats.min(axis=0),
        np.median(stats, axis=0),
        np.quantile(stats, 0.75, axis=0),
        np.quantile(stats, 0.90, axis=0),
    ], axis=1)
    return aggregated.reshape(-1)


@check_folders("tgt_folder")
def lgbm_cellstats(
    src_folders: tuple[Path, ...],
    tgt_folder: str | os.PathLike | Path,
    task: str           = "cont",
    pre: float          = 2.0,
    post: float         = 2.0,
    max_len: int        = 64,
    transform: str      = "zscore",
    split_by: str       = "trial",
    feature_set: str    = "basic",
    fixed_after: float | None = None,
    regularization: str = "default",
    drop_overlap: str   = "none",
    test_size: float    = 0.2,
    valid_size: float   = 0.2,
    n_estimators: int   = 2000,
    learning_rate: float = 0.03,
    eval_every: int     = 10,
    patience: int       = 15,
    min_class: int      = 10,
    seed: int           = 42,
) -> pd.DataFrame:
    """
    Event-level prediction with per-cell statistics aggregated over the cells and one LightGBM shared by all tokens.
    Windows, signal transform, split, class weighting, early stopping & threshold rule are the same as deepset_predict

    :param src_folders: The folders that contain the B & S csv files
    :param tgt_folder: The folder where the outputs will be placed
    :param task: The key of TASKS that decides the positive behaviors
    :param pre: The seconds before the event start
    :param post: The seconds after the event end
    :param max_len: The most seconds kept for each window
    :param transform: The per-session signal transform (see build_spans)
    :param split_by: "trial" or "event" (see deepset_predict)
    :param feature_set: "basic" (CELL_FEATURES x CELL_AGGREGATES, 70) or
                        "extended" (EXT_CELL_FEATURES x EXT_CELL_AGGREGATES + EXT_POPULATION_FEATURES, 319)
    :param fixed_after: None for the [start - pre, end + post] window, or the seconds after the start for the
                        fixed-length [start - pre, start + fixed_after] window (removes the duration cue)
    :param regularization: The key of LGBM_PRESETS
    :param drop_overlap: Which negative events are removed (from train, valid AND test) because they overlap a positive:
                         "none", "event" (the event itself overlaps a positive event in time),
                         "window25" (>= 25% of its window lies inside positive events) or
                         "window_any" (its window contains any part of a positive event)
    :param test_size: The ratio of the testing set (per token)
    :param valid_size: The ratio of the validation set inside the training set (early stopping & threshold)
    :param n_estimators: The most boosting rounds
    :param learning_rate: The LightGBM learning rate
    :param eval_every: The rounds between two evaluations / log lines
    :param patience: The evaluations without validation improvement before early stopping
    :param min_class: The least number of events of each class for a token to enter the per-token summary
    :param seed: The random seed

    :return: The per-token summary table
    """

    logger = SyncLogger(level="INFO")
    window = "variable" if fixed_after is None else f"fixed{pre:g}+{fixed_after:g}"
    tag = f"{task}|lgbm|{feature_set}|{window}|{regularization}|drop={drop_overlap}|split={split_by}"    # shown on every log line to tell the runs apart
    n_dropped = 0
    extract = event_features if feature_set == "basic" else extended_event_features
    feature_post = post if fixed_after is None else 0.0     # the fixed window has no post-end part

    # build the events of all tokens and split each token exactly as deepset_predict
    tables, features = [], []
    for folder in src_folders:
        for token in fetch_tokens(folder=Path(folder), shard="all"):
            events, wins = build_spans(
                Path(folder) / f"{token}_B.csv", Path(folder) / f"{token}_S.csv",
                pre, post, max_len, transform=transform, fixed_after=fixed_after,
            )
            y = events["behavior"].isin(TASKS[task]).to_numpy(dtype=int)

            # remove the negative events overlapping the positive ones (label cleaning, applied before the split)
            if drop_overlap != "none":
                keep = ~(y == 0) | ~overlap_mask(events, y, drop_overlap, pre, post, fixed_after)
                n_dropped += int((~keep).sum())
                events = events.loc[keep].reset_index(drop=True)
                wins = [w for w, k in zip(wins, keep) if k]
                y = y[keep]
            if split_by == "trial":
                groups = events["trial"].to_numpy()
            elif split_by == "event":
                groups = np.arange(len(events))
            else:
                raise ValueError(f"unknown split_by: {split_by}")

            side = np.full(len(y), "train", dtype=object)
            train_idx, test_idx = _split(y, groups, round(1 / test_size), seed)
            side[test_idx] = "test"
            _, sub_valid = _split(y[train_idx], groups[train_idx], round(1 / valid_size), seed)
            side[train_idx[sub_valid]] = "valid"

            tables.append(events.assign(token=token, real=y, side=side))
            features.extend(extract(w, pre, feature_post) for w in wins)

    table = pd.concat(tables, ignore_index=True)
    names = feature_names(feature_set)
    x = pd.DataFrame(np.stack(features), columns=names)
    y = table["real"].to_numpy()
    index = {s: np.flatnonzero(table["side"].to_numpy() == s) for s in ("train", "valid", "test")}
    sets = {s: (x.iloc[index[s]], y[index[s]]) for s in index}
    logger.info(
        role="lgbm_cellstats",
        message=f"[{tag}] {table['token'].nunique()} tokens | {len(names)} features | dropped negatives={n_dropped} | "
                f"train={len(index['train'])} valid={len(index['valid'])} test={len(index['test'])} | "
                f"positive rate={y.mean():.3f}"
    )

    # every eval_every rounds: threshold on the validation set, F1 on train / valid / test (test is monitored only),
    # early stopping on the validation macro F1
    history, state = [], {"best": -1.0, "best_iter": 0, "wait": 0}

    def monitor(env: lgb.callback.CallbackEnv) -> None:
        it = env.iteration + 1
        if it % eval_every:
            return
        probs = {s: env.model.predict(sets[s][0], num_iteration=it) for s in sets}
        threshold = _best_threshold(sets["valid"][1], probs["valid"])
        scores = {s: f1_scores(sets[s][1], (probs[s] >= threshold).astype(int)) for s in sets}
        score = scores["valid"]["test_f1_macro"]
        history.append({
            "iteration": it,
            "threshold": threshold,
            "valid_f1_macro": score,
            **{f"{s}_f1_{k}": scores[s][f"test_f1_{k}"] for s in sets for k in ("pos", "neg")},
        })
        logger.info(
            role="lgbm_cellstats",
            message=f"[{tag}] iter {it:4d} | valid f1_macro={score:.3f} | f1 {task}/other: "
                    f"train={scores['train']['test_f1_pos']:.3f}/{scores['train']['test_f1_neg']:.3f} "
                    f"valid={scores['valid']['test_f1_pos']:.3f}/{scores['valid']['test_f1_neg']:.3f} "
                    f"test={scores['test']['test_f1_pos']:.3f}/{scores['test']['test_f1_neg']:.3f}"
        )
        if score > state["best"]:
            state.update(best=score, best_iter=it, wait=0)
        else:
            state["wait"] += 1
            if state["wait"] >= patience:
                raise lgb.callback.EarlyStopException(state["best_iter"], [("valid", "f1_macro", state["best"], True)])

    train_y = sets["train"][1]
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample_freq=1,
        **LGBM_PRESETS[regularization],
        scale_pos_weight=(1 - train_y).sum() / max(train_y.sum(), 1),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(sets["train"][0], train_y, callbacks=[monitor])
    best_iter = state["best_iter"]

    booster = model.booster_
    threshold = _best_threshold(sets["valid"][1], booster.predict(sets["valid"][0], num_iteration=best_iter))
    logger.info(
        role="lgbm_cellstats",
        message=f"[{tag}] stopped at iter {history[-1]['iteration']} | best iter={best_iter} | "
                f"best valid macro F1={state['best']:.3f} | threshold={threshold:.2f}"
    )

    # control: predicting from the window length only, same split & threshold rule
    length = np.log(table["length"].to_numpy(dtype=float))[:, None]
    control = LogisticRegression(class_weight="balanced").fit(length[index["train"]], y[index["train"]])
    control_threshold = _best_threshold(y[index["valid"]], control.predict_proba(length[index["valid"]])[:, 1])

    test = table.iloc[index["test"]].copy()
    test["probability"] = booster.predict(sets["test"][0], num_iteration=best_iter)
    test["prediction"] = (test["probability"] >= threshold).astype(int)
    test["length_only"] = (control.predict_proba(length[index["test"]])[:, 1] >= control_threshold).astype(int)

    records = []
    for token, part in test.groupby("token", sort=False):
        events = table.loc[table["token"] == token]
        record = {"token": token, "n_events": len(events), "n_pos": int(events["real"].sum()), "n_test": len(part),
                  "test_pos": int(part["real"].sum())}
        if min(record["n_pos"], record["n_events"] - record["n_pos"]) < min_class:
            record["status"] = "skipped: too few events in one class"
        else:
            majority = int(events.loc[events["side"] != "test", "real"].mean() >= 0.5)
            record.update({
                **_metrics(part["real"].to_numpy(), part["prediction"].to_numpy()),
                "majority_accuracy": accuracy_score(part["real"], np.full(len(part), majority)),
                "status": "ok",
            })
        records.append(record)
    summary = pd.DataFrame(records)

    pooled = pd.DataFrame([
        {"model": "lgbm_cellstats", **_metrics(test["real"].to_numpy(), test["prediction"].to_numpy())},
        {"model": "length_only", **_metrics(test["real"].to_numpy(), test["length_only"].to_numpy())},
        {"model": "majority", **_metrics(test["real"].to_numpy(), np.zeros(len(test), dtype=int))},
    ])
    importance = pd.DataFrame({
        "feature": names,
        "gain": booster.feature_importance(importance_type="gain", iteration=best_iter),
    }).sort_values("gain", ascending=False)

    summary.to_csv(Path(tgt_folder) / "summary.csv", index=False)
    split_counts = (
        table.assign(column=table["side"] + "_" + np.where(table["real"] == 1, "reach", "other"))
        .groupby(["token", "column"]).size().unstack(fill_value=0)
        .reindex(index=table["token"].unique(),
                 columns=[f"{s}_{c}" for s in ("train", "valid", "test") for c in ("reach", "other")], fill_value=0)
    )
    split_counts.to_csv(Path(tgt_folder) / "split_counts.csv")
    pooled.to_csv(Path(tgt_folder) / "pooled.csv", index=False)
    test.to_csv(Path(tgt_folder) / "raw.csv", index=False)
    pd.DataFrame(history).to_csv(Path(tgt_folder) / "history.csv", index=False)
    importance.to_csv(Path(tgt_folder) / "importance.csv", index=False)
    booster.save_model(str(Path(tgt_folder) / "model.txt"), num_iteration=best_iter)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        test["real"], test["prediction"], display_labels=["Other", task],
        cmap="Reds", im_kw={"vmin": 0.0, "vmax": 1.0}, normalize="true", ax=ax,
    )
    plt.savefig(Path(tgt_folder) / "matrix.pdf")
    plt.close()

    logger.info(role="lgbm_cellstats", message=f"[{tag}] top features by gain: " +
                ", ".join(importance["feature"].head(10)))
    logger.info(role="lgbm_cellstats", message=f"[{tag}] outputs saved to {tgt_folder}")
    return summary
