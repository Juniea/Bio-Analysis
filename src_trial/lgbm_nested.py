import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time
import warnings

import lightgbm as lgb
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

from tools import SyncLogger, check_folders, fetch_tokens
from .trial_predict import TASKS, f1_scores
from .cell_select import sps_select, sps_fixed_select, rfecv_select
from .deepset_predict import build_spans
from .lgbm_cellstats import (
    LGBM_PRESETS, context_mask, event_features, extended_event_features, feature_names, first_negatives, overlap_mask,
)


def _assign_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """
    Assign every event of one token to one of n_splits folds (stratified when both classes are frequent enough)

    :return: fold id [n_events]
    """
    folds = np.zeros(len(y), dtype=int)
    n_groups = len(np.unique(groups))
    if np.bincount(y, minlength=2).min() >= n_splits and n_groups >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y, groups)
    else:
        splitter = GroupKFold(n_splits=min(n_splits, n_groups)).split(y, y, groups)
    for k, (_, idx) in enumerate(splitter):
        folds[idx] = k
    return folds


def _scores(real: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    """
    Threshold-based scores plus the threshold-free ROC-AUC (NaN when only one class is present)
    """
    two_classes = len(np.unique(real)) > 1
    return {
        "accuracy": accuracy_score(real, pred),
        "balanced_accuracy": balanced_accuracy_score(real, pred) if two_classes else np.nan,
        "roc_auc": roc_auc_score(real, prob) if two_classes else np.nan,
        **{k.replace("test_", ""): v for k, v in f1_scores(real, pred).items()},
    }


def sample_weights(tokens: np.ndarray, y: np.ndarray, scheme: str, min_count: int = 5) -> np.ndarray:
    """
    The per-event training weights (normalized to mean 1)

    :param tokens: The token of every training event
    :param y: The label of every training event
    :param scheme: "global" (positives weighted by #neg / #pos over the whole training set) or
                   "session_balanced" (every token gets the same total weight, split half / half between its classes)
    :param min_count: The least class count used in the session-balanced weights (avoids huge weights for rare classes)

    :return: weights [n_events]
    """
    if scheme == "global":
        w = np.where(y == 1, (1 - y).sum() / max(y.sum(), 1), 1.0)
    elif scheme == "session_balanced":
        w = np.zeros(len(y))
        for token in np.unique(tokens):
            in_token = tokens == token
            present = [c for c in (0, 1) if (in_token & (y == c)).any()]
            for c in present:
                mask = in_token & (y == c)
                w[mask] = 1.0 / (len(present) * max(mask.sum(), min_count))
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    return w / w.mean()


def session_balanced_scores(real: np.ndarray, prob: np.ndarray, pred: np.ndarray, tokens: np.ndarray) -> dict:
    """
    The pooled scores with the session prior removed: every token weighs the same in total, half of it on its reaches
    and half on its non-reaches (the tokens with one class only weigh nothing), so the token's reach rate alone scores
    at chance; the probability (AUC) and the prediction (F1, balanced accuracy) are still those of the one model

    :return: the scores plus the number of tokens that weigh
    """
    weight = np.zeros(len(real))
    for token in np.unique(tokens):
        own = tokens == token
        if len(np.unique(real[own])) < 2:
            continue
        for c in (0, 1):
            idx = own & (real == c)
            weight[idx] = 0.5 / idx.sum()
    kept = weight > 0
    neg, pos = f1_score(real, pred, labels=[0, 1], average=None, zero_division=0, sample_weight=weight)
    return {
        "accuracy": accuracy_score(real, pred, sample_weight=weight),
        "balanced_accuracy": balanced_accuracy_score(real, pred, sample_weight=weight),
        "roc_auc": roc_auc_score(real[kept], prob[kept], sample_weight=weight[kept]),
        "f1_macro": (neg + pos) / 2, "f1_pos": pos, "f1_neg": neg,
        "tokens": int(len(np.unique(tokens[kept]))),
    }


def _best_threshold_curve(real: np.ndarray, prob: np.ndarray, metric: str = "f1_macro") -> tuple[float, float]:
    """
    The (threshold, score) maximizing the score over the 0.05..0.95 grid

    :param metric: "f1_macro" (macro F1) or "f1_pos" (reach F1)
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_scores(real, (prob >= g).astype(int))[f"test_{metric}"] for g in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


SELECTIVITY_FEATURES = (
    "sel_lda", "sel_weighted", "sel_weighted_delta", "sel_corr_diff", "sel_dist_diff",
    "sel_top_reach", "sel_top_other", "sel_top_diff",
)
# the most frequent non-reach behaviors left after the window_any cleaning, each with its own reach-vs-X template
# ("noise generation" used to be here; it is the rig's white-noise marker, now in trial_predict.EXCLUDED)
CONTRASTS = {
    "both_in_vain": ("both in vain",),
    "walk": ("walk",),
    "cont_in_vain": ("cont in vain",),
}
EXTENDED_SELECTIVITY_FEATURES = (
    SELECTIVITY_FEATURES
    + tuple(f"sel_time_{t:+d}" for t in range(-2, 5))
    + tuple(f"sel_vs_{name}{suffix}" for name in CONTRASTS for suffix in ("", "_delta"))
    + ("sel_logreg", "sel_shrink_lda")
)
MULTI_CLASSES = ("reach", *CONTRASTS, "rest")
DECODER_SELECTIVITY_FEATURES = (
    EXTENDED_SELECTIVITY_FEATURES
    + tuple(f"sel_time_logreg_{t:+d}" for t in range(-2, 5))
    + tuple(f"sel_multi_{name}" for name in MULTI_CLASSES)
    + ("sel_logreg_cv",)
)
LOGREG_CV_GRID = (0.005, 0.01, 0.05, 0.1, 0.5)
SUBTYPE_FEATURES = ("sel_pure", "sel_pure_delta", "sel_mixed", "sel_mixed_delta", "sel_sub_pure", "sel_sub_mixed", "sel_sub_other")
GLM_FEATURES = ("sel_glm_proj", "sel_glm_contrast", "sel_glm_lda")
UNMIX_FEATURES = (
    "unmix_reach_0", "unmix_reach_near", "unmix_other_0", "unmix_reach_ratio", "unmix_other_around", "unmix_resid",
    "unmix_clean_contrast", "unmix_clean_lda",
)
SUBTYPE_GLM_FEATURES = EXTENDED_SELECTIVITY_FEATURES + SUBTYPE_FEATURES + GLM_FEATURES
EXTENDED_GLM_FEATURES = EXTENDED_SELECTIVITY_FEATURES + GLM_FEATURES
EXTENDED_UNMIX_FEATURES = EXTENDED_GLM_FEATURES + UNMIX_FEATURES


def window_feature_names(names: tuple[str, ...], pre: float, n_points: int) -> tuple[str, ...]:
    """
    Replace the default 7 sel_time_* names (window -2..+4) by one name per point of the actual window
    """
    n_pre = int(round(pre))
    times = tuple(f"sel_time_{t:+d}" for t in range(-n_pre, n_points - n_pre))
    out, inserted = [], False
    for name in names:
        if name.startswith("sel_time_") and not name.startswith("sel_time_logreg_"):
            if not inserted:
                out.extend(times)
                inserted = True
            continue
        out.append(name)
    return tuple(out)


def _unmix_dictionary(kernels: np.ndarray, n_pre: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Every GLM kernel placed at every onset shift (seconds from the window anchor) that still touches the window

    :param kernels: [classes, cells, points] onset kernels, lag = point index - n_pre
    :return: (atoms [n_atoms, cells * points], atom classes [n_atoms], atom shifts [n_atoms])
    """
    n_classes, n_cells, n_points = kernels.shape
    offsets = np.arange(n_points) - n_pre
    atoms, classes, shifts = [], [], []
    for shift in range(offsets[0] - offsets[-1], offsets[-1] - offsets[0] + 1):
        lags = offsets - shift
        inside = (lags >= offsets[0]) & (lags <= offsets[-1])
        for c in range(n_classes):
            atom = np.zeros((n_cells, n_points))
            atom[:, inside] = kernels[c][:, lags[inside] - offsets[0]]
            atoms.append(atom.reshape(-1))
            classes.append(c)
            shifts.append(shift)
    return np.array(atoms), np.array(classes), np.array(shifts)


def _apply_unmix(extra: dict, wins: np.ndarray) -> np.ndarray:
    """
    Explain every window (no annotation used) as a non-negative sum of the GLM kernels at all onset shifts, then read
    the reach / other amounts at the anchor (shift 0) and the window cleaned from the other shifts

    :return: [n_events, len(UNMIX_FEATURES)]
    """
    atoms, classes, shifts = extra["unmix_atoms"], extra["unmix_classes"], extra["unmix_shifts"]
    design = np.vstack([atoms.T, np.sqrt(extra["unmix_ridge"]) * np.eye(len(atoms))])
    reach, other = extra["glm_reach"], extra["glm_other"]
    diff, mid = reach - other, (reach + other) / 2
    anchor, reach_atom = shifts == 0, classes == 0
    out = np.full((len(wins), len(UNMIX_FEATURES)), np.nan)
    for i, w in enumerate(np.nan_to_num(wins)):
        target = (w - extra["glm_intercept"][:, None]).reshape(-1)
        coef, _ = nnls(design, np.concatenate([target, np.zeros(len(atoms))]), maxiter=50 * len(atoms))
        reach_0 = coef[anchor & reach_atom].sum()
        other_0 = coef[anchor & ~reach_atom].sum()
        residual = target - atoms.T @ coef
        cleaned = (target - atoms[~anchor].T @ coef[~anchor]).reshape(w.shape)
        out[i] = [
            reach_0,
            coef[reach_atom & (np.abs(shifts) <= 1)].max(),
            other_0,
            reach_0 / (reach_0 + other_0 + 1e-6),
            coef[~anchor].sum(),
            np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-6),
            ((cleaned - mid) * diff).sum() / max((diff ** 2).sum(), 1e-6),
            ((cleaned - mid) * diff / extra["glm_resvar"][:, None]).sum(),
        ]
    return out


def _fit_subtypes_glm(
    wins: np.ndarray,
    y: np.ndarray,
    behaviors: np.ndarray,
    pre: float,
    extra: dict,
    logreg_c: float,
    min_count: int = 3,
    ridge: float = 10.0,
    eps: float = 1e-2,
) -> dict:
    """
    Reach split into pure (no other event inside the window) & mixed reaches, each with its own reach-vs-other
    template plus a 3-class decoder; and a ridge GLM whose per-behavior onset kernels separate the pure reach response
    from the overlapping events (learned from these fit events' labels only)

    :param extra: {"pure": [n_events] bool, "starts": [n_events] fit event starts,
                   "other_starts": starts of the token events outside this fit set (their labels are unknown),
                   "subtypes": whether to learn the pure / mixed reach part (False keeps the GLM only)}
    """
    n_pre = int(round(pre))
    out = {"subtypes": {}, "with_subtypes": extra["subtypes"]}
    if extra["subtypes"]:
        resp, delta = _resp_delta(wins, n_pre)
        pos, neg = y == 1, y == 0
        pure, mixed = pos & extra["pure"], pos & ~extra["pure"]
        for name, group in (("pure", pure), ("mixed", mixed)):
            if group.sum() >= min_count:
                out["subtypes"][name] = [(((mu1 + mu0) / 2), sd, (mu1 - mu0) / sd)
                                         for mu1, mu0, sd in (_contrast_stats(resp, group, neg, eps), _contrast_stats(delta, group, neg, eps))]

        flat = np.nan_to_num(wins.reshape(len(wins), -1))
        scaler = StandardScaler().fit(flat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["sub_scaler"] = scaler
            out["sub_model"] = LogisticRegression(C=logreg_c, class_weight="balanced", max_iter=1000).fit(
                scaler.transform(flat), np.where(pure, 0, np.where(mixed, 1, 2)))

    # GLM: activity(cell, time) = intercept + sum over the nearby fit events of kernel[class](lag); the time points
    # near an event outside the fit set are skipped since its behavior is unknown
    n_points = wins.shape[2]
    offsets = np.arange(n_points) - n_pre
    classes = _multi_labels(y, behaviors)
    starts, other = extra["starts"], extra["other_starts"]
    xs, ys = [], []
    for i in range(len(wins)):
        tau = starts[i] + offsets
        lags = np.rint(tau[:, None] - starts[None, :]).astype(int)
        near = (lags >= offsets[0]) & (lags <= offsets[-1])
        x = np.zeros((n_points, len(MULTI_CLASSES) * n_points))
        for t, e in zip(*np.nonzero(near)):
            x[t, classes[e] * n_points + lags[t, e] - offsets[0]] += 1
        keep = np.ones(n_points, dtype=bool)
        if len(other):
            unknown = np.rint(tau[:, None] - other[None, :])
            keep = ~((unknown >= offsets[0]) & (unknown <= offsets[-1])).any(axis=1)
        xs.append(x[keep])
        ys.append(np.nan_to_num(wins[i].T[keep]))
    x, target = np.concatenate(xs), np.concatenate(ys)
    x = np.hstack([np.ones((len(x), 1)), x])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ target)  # [1 + classes * points, cells]
    kernels = coef[1:].reshape(len(MULTI_CLASSES), n_points, -1).transpose(0, 2, 1)  # [classes, cells, points]
    out["glm_intercept"] = coef[0]
    out["glm_reach"] = kernels[0]
    out["glm_other"] = kernels[1:].mean(axis=0)
    out["glm_resvar"] = (target - x @ coef).var(axis=0) + eps
    out["with_unmix"] = extra.get("unmix", False)
    if out["with_unmix"]:
        out["unmix_atoms"], out["unmix_classes"], out["unmix_shifts"] = _unmix_dictionary(kernels, n_pre)
        # a light ridge on the amounts, scaled to the typical atom energy
        out["unmix_ridge"] = 0.1 * float(np.median((out["unmix_atoms"] ** 2).sum(axis=1)))
    return out


def _apply_subtypes_glm(extra: dict, wins: np.ndarray, pre: float) -> np.ndarray:
    """
    :return: [n_events, len(SUBTYPE_FEATURES) + len(GLM_FEATURES)], or [n_events, len(GLM_FEATURES)] without subtypes
    """
    parts = []
    if extra["with_subtypes"]:
        resp, delta = _resp_delta(wins, int(round(pre)))
        sub = np.full((len(wins), 4), np.nan)
        for c, name in enumerate(("pure", "mixed")):
            if name in extra["subtypes"]:
                (r_mid, r_sd, r_dp), (d_mid, d_sd, d_dp) = extra["subtypes"][name]
                sub[:, 2 * c] = _weighted_score(resp, r_mid, r_sd, r_dp)
                sub[:, 2 * c + 1] = _weighted_score(delta, d_mid, d_sd, d_dp)
        probs = np.full((len(wins), 3), np.nan)
        model = extra["sub_model"]
        probs[:, model.classes_] = model.predict_proba(extra["sub_scaler"].transform(np.nan_to_num(wins.reshape(len(wins), -1))))
        parts += [sub, probs]

    centered = np.nan_to_num(wins) - extra["glm_intercept"][None, :, None]
    reach, other = extra["glm_reach"], extra["glm_other"]
    diff = reach - other
    glm = np.stack([
        (centered * reach).sum(axis=(1, 2)) / max((reach ** 2).sum(), 1e-6),
        ((centered - (reach + other) / 2) * diff).sum(axis=(1, 2)) / max((diff ** 2).sum(), 1e-6),
        ((centered - (reach + other) / 2) * diff / extra["glm_resvar"][None, :, None]).sum(axis=(1, 2)),
    ], axis=1)
    parts.append(glm)
    if extra["with_unmix"]:
        parts.append(_apply_unmix(extra, wins))
    return np.concatenate(parts, axis=1)


def _multi_labels(y: np.ndarray, behaviors: np.ndarray) -> np.ndarray:
    """
    The MULTI_CLASSES index of every event: reach, one of the CONTRASTS behaviors, or the rest
    """
    labels = np.full(len(y), len(MULTI_CLASSES) - 1)
    for c, members in enumerate(CONTRASTS.values(), start=1):
        labels[np.isin(behaviors, members)] = c
    labels[y == 1] = 0
    return labels


def _fit_decoders(
    wins: np.ndarray, y: np.ndarray, behaviors: np.ndarray, logreg_c: float, cv_folds: int = 3, seed: int = 42,
) -> dict:
    """
    The extra per-token decoders: one logistic regression per time point (cells only), a multinomial logistic
    regression over MULTI_CLASSES, and a logistic regression whose C is chosen by an inner cross-validation
    """
    decoders = {"time": []}
    wins = np.nan_to_num(wins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(wins.shape[2]):
            scaler = StandardScaler().fit(wins[:, :, t])
            model = LogisticRegression(C=logreg_c, class_weight="balanced", max_iter=500).fit(scaler.transform(wins[:, :, t]), y)
            decoders["time"].append((scaler, model))

        flat = wins.reshape(len(wins), -1)
        scaler = StandardScaler().fit(flat)
        xs = scaler.transform(flat)
        decoders["flat_scaler"] = scaler
        decoders["multi"] = LogisticRegression(C=logreg_c, class_weight="balanced", max_iter=1000).fit(
            xs, _multi_labels(y, behaviors))

        # the C with the best cross-validated AUC inside these events only (the fixed C when a class is too rare)
        best_c = logreg_c
        if min(y.sum(), (1 - y).sum()) >= cv_folds:
            folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            aucs = []
            for c in LOGREG_CV_GRID:
                prob = np.zeros(len(y))
                for fit, hold in folds.split(xs, y):
                    model = LogisticRegression(C=c, class_weight="balanced", max_iter=500).fit(xs[fit], y[fit])
                    prob[hold] = model.decision_function(xs[hold])
                aucs.append(roc_auc_score(y, prob))
            best_c = LOGREG_CV_GRID[int(np.argmax(aucs))]
        decoders["logreg_cv"] = LogisticRegression(C=best_c, class_weight="balanced", max_iter=500).fit(xs, y)
        decoders["logreg_cv_c"] = best_c
    return decoders


def _apply_decoders(decoders: dict, wins: np.ndarray) -> np.ndarray:
    """
    :return: [n_events, len(DECODER_SELECTIVITY_FEATURES) - len(EXTENDED_SELECTIVITY_FEATURES)]
    """
    wins = np.nan_to_num(wins)
    time_scores = np.stack([model.decision_function(scaler.transform(wins[:, :, t]))
                            for t, (scaler, model) in enumerate(decoders["time"])], axis=1)
    xs = decoders["flat_scaler"].transform(wins.reshape(len(wins), -1))
    multi = np.full((len(wins), len(MULTI_CLASSES)), np.nan)
    multi[:, decoders["multi"].classes_] = decoders["multi"].predict_proba(xs)
    return np.concatenate([time_scores, multi, decoders["logreg_cv"].decision_function(xs)[:, None]], axis=1)


def _contrast_stats(v: np.ndarray, pos: np.ndarray, neg: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (positive mean, negative mean, pooled std) of every column of v
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu1, mu0 = np.nanmean(v[pos], axis=0), np.nanmean(v[neg], axis=0)
        n = np.isfinite(v[pos]).sum(axis=0) + np.isfinite(v[neg]).sum(axis=0)
        var = (np.nansum((v[pos] - mu1) ** 2, axis=0) + np.nansum((v[neg] - mu0) ** 2, axis=0)) / np.maximum(n - 2, 1)
    # columns masked in a whole class carry no selectivity
    mu1, mu0 = np.nan_to_num(mu1), np.nan_to_num(mu0)
    return mu1, mu0, np.sqrt(var + eps)


def _weighted_score(v: np.ndarray, mid: np.ndarray, sd: np.ndarray, dprime: np.ndarray) -> np.ndarray:
    """
    The d'-weighted standardized response, summed over the last axis (masked NaN entries are skipped)
    """
    kept = np.isfinite(v)
    den = (np.abs(dprime) * kept).sum(axis=-1)
    num = np.where(kept, (v - mid) / sd * dprime, 0.0).sum(axis=-1)
    with np.errstate(all="ignore"):
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)


def _resp_delta(wins: np.ndarray, n_pre: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The per-cell mean after the start & its difference from the mean before the start, skipping the masked points

    :return: ([n_events, n_cells], [n_events, n_cells])
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        resp = np.nanmean(wins[:, :, n_pre:], axis=2)
        return resp, resp - np.nanmean(wins[:, :, :n_pre], axis=2)


def _lda_raw(flat: np.ndarray, mid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    The diagonal LDA projection, the masked points contributing nothing
    """
    return np.where(np.isfinite(flat), (flat - mid) * weights, 0.0).sum(axis=1)


def _template(
    wins: np.ndarray,
    y: np.ndarray,
    pre: float,
    behaviors: np.ndarray | None = None,
    extended: bool = False,
    decoders: bool = False,
    extra: dict | None = None,
    top_frac: float = 0.1,
    min_contrast: int = 3,
    logreg_c: float = 0.05,
    eps: float = 1e-2,
) -> dict | None:
    """
    The per-cell reach selectivity of one token, learned from its training events only

    :param wins: [n_events, n_cells, n_points] fixed windows of one token
    :param y: [n_events] 1 = reach
    :param behaviors: [n_events] behavior names, needed by the per-behavior contrasts (extended only)
    :param extended: Whether to add the time-resolved, per-behavior contrast & linear decoder selectivity
    :param decoders: Whether to add the extra decoders of _fit_decoders (extended only)
    :param extra: The pure / mixed reach & GLM context of these events (see _fit_subtypes_glm, extended only)

    :return: None when either class has fewer than 2 events
    """
    if y.sum() < 2 or (1 - y).sum() < 2:
        return None
    n_pre = int(round(pre))
    flat = wins.reshape(len(wins), -1)
    resp, delta = _resp_delta(wins, n_pre)
    pos, neg = y == 1, y == 0

    f_mu1, f_mu0, f_sd = _contrast_stats(flat, pos, neg, eps)
    r_mu1, r_mu0, r_sd = _contrast_stats(resp, pos, neg, eps)
    d_mu1, d_mu0, d_sd = _contrast_stats(delta, pos, neg, eps)
    template = {
        "lda_w": (f_mu1 - f_mu0) / f_sd ** 2, "lda_mid": (f_mu1 + f_mu0) / 2,
        "r_mu1": r_mu1, "r_mu0": r_mu0, "r_sd": r_sd, "r_dprime": (r_mu1 - r_mu0) / r_sd,
        "d_mid": (d_mu1 + d_mu0) / 2, "d_sd": d_sd, "d_dprime": (d_mu1 - d_mu0) / d_sd,
        "pre": pre, "extended": extended,
    }
    k = max(1, int(round(top_frac * wins.shape[1])))
    order = np.argsort(template["r_dprime"])
    template["top"], template["bottom"] = order[-k:], order[:k]
    lda = _lda_raw(flat, template["lda_mid"], template["lda_w"])
    template["lda_center"], template["lda_scale"] = lda.mean(), max(lda.std(), 1e-6)
    if not extended:
        return template

    # time-resolved selectivity: one d' per cell & time point, cells x points
    shape = wins.shape[1:]
    template["t_mid"] = ((f_mu1 + f_mu0) / 2).reshape(shape)
    template["t_sd"] = f_sd.reshape(shape)
    template["t_dprime"] = ((f_mu1 - f_mu0) / f_sd).reshape(shape)

    # reach vs one confusable behavior at a time
    template["contrasts"] = {}
    for name, members in CONTRASTS.items():
        other = neg & np.isin(behaviors, members)
        if other.sum() < min_contrast:
            continue
        c_r = _contrast_stats(resp, pos, other, eps)
        c_d = _contrast_stats(delta, pos, other, eps)
        template["contrasts"][name] = [(((mu1 + mu0) / 2), sd, (mu1 - mu0) / sd) for mu1, mu0, sd in (c_r, c_d)]

    # linear decoders over all cells x points (they use the correlations between cells);
    # the masked points take the training mean, so they add nothing after the standardization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        template["fill"] = np.nan_to_num(np.nanmean(flat, axis=0))
    x = np.where(np.isfinite(flat), flat, template["fill"])
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        template["scaler"] = scaler
        template["logreg"] = LogisticRegression(C=logreg_c, class_weight="balanced", max_iter=500).fit(xs, y)
        template["shrink_lda"] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(xs, y)
    if decoders:
        template["decoders"] = _fit_decoders(wins, y, behaviors, logreg_c)
    if extra is not None:
        template["subtypes_glm"] = _fit_subtypes_glm(wins, y, behaviors, pre, extra, logreg_c)
    return template


def _apply_template(template: dict, wins: np.ndarray) -> np.ndarray:
    """
    The selectivity features of some events of the same token

    :return: [n_events, len(SELECTIVITY_FEATURES)] or [n_events, len(EXTENDED_SELECTIVITY_FEATURES)]
    """
    n_pre = int(round(template["pre"]))
    flat = wins.reshape(len(wins), -1)
    resp, delta = _resp_delta(wins, n_pre)

    lda = (_lda_raw(flat, template["lda_mid"], template["lda_w"]) - template["lda_center"]) / template["lda_scale"]
    z = (resp - (template["r_mu1"] + template["r_mu0"]) / 2) / template["r_sd"]
    weighted = _weighted_score(resp, (template["r_mu1"] + template["r_mu0"]) / 2, template["r_sd"], template["r_dprime"])
    weighted_delta = _weighted_score(delta, template["d_mid"], template["d_sd"], template["d_dprime"])

    def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean()
        with np.errstate(all="ignore"):
            return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))

    corr_diff = row_corr(resp, template["r_mu1"]) - row_corr(resp, template["r_mu0"])
    n_cells = resp.shape[1]
    dist_diff = (np.linalg.norm((resp - template["r_mu0"]) / template["r_sd"], axis=1)
                 - np.linalg.norm((resp - template["r_mu1"]) / template["r_sd"], axis=1)) / np.sqrt(n_cells)
    top_reach, top_other = z[:, template["top"]].mean(axis=1), z[:, template["bottom"]].mean(axis=1)
    basic = np.stack([lda, weighted, weighted_delta, corr_diff, dist_diff, top_reach, top_other, top_reach - top_other], axis=1)
    if not template["extended"]:
        return basic

    # time-resolved: the d'-weighted response at every time point, summed over the cells
    time_scores = _weighted_score(
        wins.transpose(0, 2, 1), template["t_mid"].T, template["t_sd"].T, template["t_dprime"].T,
    )
    contrasts = np.full((len(wins), 2 * len(CONTRASTS)), np.nan)
    for c, name in enumerate(CONTRASTS):
        if name in template["contrasts"]:
            (r_mid, r_sd, r_dp), (d_mid, d_sd, d_dp) = template["contrasts"][name]
            contrasts[:, 2 * c] = _weighted_score(resp, r_mid, r_sd, r_dp)
            contrasts[:, 2 * c + 1] = _weighted_score(delta, d_mid, d_sd, d_dp)
    xs = template["scaler"].transform(np.where(np.isfinite(flat), flat, template["fill"]))
    decoders = np.stack([
        template["logreg"].decision_function(xs), template["shrink_lda"].decision_function(xs),
    ], axis=1)
    features = np.concatenate([basic, time_scores, contrasts, decoders], axis=1)
    if "decoders" in template:
        features = np.concatenate([features, _apply_decoders(template["decoders"], wins)], axis=1)
    if "subtypes_glm" in template:
        features = np.concatenate([features, _apply_subtypes_glm(template["subtypes_glm"], wins, template["pre"])], axis=1)
    return features


def selectivity_features(
    windows: list[np.ndarray],
    tokens: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    pre: float,
    behaviors: np.ndarray | None = None,
    extended: bool = False,
    n_splits: int = 5,
    seed: int = 42,
    fit_windows: list[np.ndarray] | None = None,
    decoders: bool = False,
    extra_ctx: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reach selectivity features without label leakage: the apply events use the template of all fit events of their
    token, while every fit event uses a template learned without its own cross-fitting fold

    :param behaviors: [n_events] behavior names of all events, needed when extended
    :param extended: Whether to return EXTENDED_SELECTIVITY_FEATURES instead of SELECTIVITY_FEATURES
    :param fit_windows: The windows read for the fit events (e.g. with the context masked); None uses windows
    :param decoders: Whether to return DECODER_SELECTIVITY_FEATURES (extended only)
    :param extra_ctx: {"starts": [n_events] start seconds, "pure": [n_events] bool no other event inside the window}
                      of all events plus "subtypes" (bool), to return SUBTYPE_GLM_FEATURES, or EXTENDED_GLM_FEATURES
                      without the subtypes (extended only)

    :return: ([len(fit_idx), n_features], [len(apply_idx), n_features]), NaN when a token lacks one class
    """
    if extra_ctx is not None:
        names = (SUBTYPE_GLM_FEATURES if extra_ctx["subtypes"] else EXTENDED_GLM_FEATURES) \
            + (UNMIX_FEATURES if extra_ctx.get("unmix", False) else ())
    else:
        names = DECODER_SELECTIVITY_FEATURES if decoders else EXTENDED_SELECTIVITY_FEATURES if extended else SELECTIVITY_FEATURES
    names = window_feature_names(names, pre, windows[fit_idx[0]].shape[1])
    n_features = len(names)
    fit_out = np.full((len(fit_idx), n_features), np.nan)
    apply_out = np.full((len(apply_idx), n_features), np.nan)
    for token in np.unique(tokens[fit_idx]):
        f_pos = np.flatnonzero(tokens[fit_idx] == token)
        a_pos = np.flatnonzero(tokens[apply_idx] == token)
        f_wins = np.stack([(windows if fit_windows is None else fit_windows)[i] for i in fit_idx[f_pos]])
        f_y = y[fit_idx[f_pos]]
        f_beh = behaviors[fit_idx[f_pos]] if behaviors is not None else None
        token_all = np.flatnonzero(tokens == token)

        def context(fit_global: np.ndarray) -> dict | None:
            # the events of the token outside the fit set only contribute their start times, never their labels
            if extra_ctx is None:
                return None
            return {"pure": extra_ctx["pure"][fit_global], "starts": extra_ctx["starts"][fit_global],
                    "other_starts": extra_ctx["starts"][np.setdiff1d(token_all, fit_global)],
                    "subtypes": extra_ctx["subtypes"], "unmix": extra_ctx.get("unmix", False)}

        template = _template(f_wins, f_y, pre, f_beh, extended, decoders, context(fit_idx[f_pos]))
        if template is not None and len(a_pos):
            apply_out[a_pos] = _apply_template(template, np.stack([windows[i] for i in apply_idx[a_pos]]))

        folds = _assign_folds(f_y, np.arange(len(f_y)), n_splits, seed)
        for k in np.unique(folds):
            inner = _template(
                f_wins[folds != k], f_y[folds != k], pre,
                f_beh[folds != k] if f_beh is not None else None, extended, decoders,
                context(fit_idx[f_pos][folds != k]),
            )
            if inner is not None:
                fit_out[f_pos[folds == k]] = _apply_template(inner, f_wins[folds == k])
    return fit_out, apply_out


def _f1_triplet(scores: dict) -> str:
    """
    "reach F1/non-reach F1/macro F1" of an f1_scores result
    """
    return f"{scores['test_f1_pos']:.3f}/{scores['test_f1_neg']:.3f}/{scores['test_f1_macro']:.3f}"


def _make_model(
    n_estimators: int, learning_rate: float, regularization: str, y_train: np.ndarray, seed: int, weighting: str = "global",
):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample_freq=1,
        **LGBM_PRESETS[regularization],
        # with session-balanced sample weights the classes are already balanced inside every token
        scale_pos_weight=(1 - y_train).sum() / max(y_train.sum(), 1) if weighting == "global" else 1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


@check_folders("tgt_folder")
def lgbm_nested(
    src_folders: tuple[Path, ...],
    tgt_folder: str | os.PathLike | Path,
    task: str           = "cont",
    pre: float          = 2.0,
    post: float         = 2.0,
    max_len: int        = 64,
    transform: str      = "zscore",
    split_by: str       = "event",
    feature_set: str    = "extended",
    fixed_after: float | None = 4.0,
    regularization: str = "regularized",
    drop_overlap: str   = "window_any",
    outer_folds: int    = 5,
    inner_folds: int    = 5,
    max_trees: int      = 1000,
    tree_step: int      = 10,
    learning_rate: float = 0.03,
    drop_prefixes: tuple[str, ...] = (),
    selectivity: str    = "none",
    log_every: int      = 100,
    mask_context: str   = "none",
    weighting: str      = "global",
    cell_selection: str = "none",
    circular_shift: int | None = None,
    stages: tuple[str, ...] | None = None,
    train_only_stages: tuple[str, ...] = (),
    select_by: str      = "f1_macro",
    n_blocks: int       = 10,
    feature_window: tuple[float, float] | None = None,
    clean_window: tuple[float, float] | None = None,
    seed: int           = 42,
) -> pd.DataFrame:
    """
    Nested cross-validation of the per-cell statistics + LightGBM pipeline (see lgbm_cellstats):
    the outer folds give every event exactly one testing prediction, the inner folds (inside the outer training part
    only) choose the number of trees and the decision threshold by the out-of-fold macro F1

    :param src_folders: The folders that contain the B & S csv files
    :param tgt_folder: The folder where the outputs will be placed
    :param task: The key of TASKS that decides the positive behaviors
    :param pre: The seconds before the event start
    :param post: The seconds after the event end (variable window only)
    :param max_len: The most seconds kept for each window
    :param transform: The per-session signal transform (see build_spans)
    :param split_by: "trial", "event" or "block" (n_blocks contiguous time blocks per token, against slow drift),
                     how the events of one token are assigned to the folds
    :param feature_set: "basic" or "extended" (see lgbm_cellstats)
    :param fixed_after: None for the [start - pre, end + post] window, or the seconds after the start for a fixed window
    :param regularization: The key of LGBM_PRESETS
    :param drop_overlap: The negative-event cleaning rule (see lgbm_cellstats), or "symmetric": also drop the reaches
                         whose window touches a cont in vain; a "_dedup" suffix (e.g. "window_any_dedup") then keeps
                         only the first of the overlapping non-reach windows (see first_negatives)
    :param outer_folds: The number of outer (testing) folds
    :param inner_folds: The number of inner (selection) folds inside each outer training part
    :param max_trees: The most boosting rounds considered by the inner selection
    :param tree_step: The spacing of the candidate numbers of trees
    :param learning_rate: The LightGBM learning rate
    :param drop_prefixes: The feature name prefixes removed from the hand-crafted features, e.g. ("min_",)
    :param selectivity: "none", "basic" (SELECTIVITY_FEATURES), "extended" (EXTENDED_SELECTIVITY_FEATURES) or
                        "decoders" (DECODER_SELECTIVITY_FEATURES), "subtypes_glm" (SUBTYPE_GLM_FEATURES) or
                        "glm" (EXTENDED_GLM_FEATURES) or "glm_unmix" (EXTENDED_UNMIX_FEATURES),
                        the per-token reach selectivity features (fixed window only)
    :param log_every: Print the training / holdout / testing F1 every log_every trees (monitoring only)
    :param mask_context: The window points covered by other events (see context_mask, fixed window & extended features
                         only): "none" keeps them, "all" masks them for every event, "train" masks them only for the
                         events used to fit (features, templates & models), the validation & testing events stay intact;
                         "train+selectivity" also masks them in the selectivity features of the validation & testing
                         events (their hand-crafted features stay intact)
    :param weighting: "global" (scale_pos_weight over the whole training part) or "session_balanced" (sample weights:
                      every token the same total weight, split half / half between its classes)
    :param cell_selection: "none", "sps" (adaptive threshold), "sps_fixed" (fixed macro F1 threshold 0.6) or "rfecv":
                           keep only the cells chosen by tianyi's SPS-SELECT / RFECV-SELECT (see cell_select), run per
                           token on the outer training events of each outer fold; the hand-crafted & selectivity
                           features are then computed from the kept cells only
    :param circular_shift: None for the real data, or the seed of the circular-shift control (see build_spans): every
                           session's signal is rolled in time by its own random offset, the events & labels stay intact
    :param stages: None for all the tokens, or the stages kept for training & testing, e.g. ("D1", "D2", "D7", "D8")
    :param train_only_stages: The stages whose tokens only join the training part of every fold (outer fold -1), never
                              tested; the scores & outputs then cover the tested events only
    :param select_by: The inner out-of-fold score that chooses the number of trees & the threshold: "f1_macro" (macro
                      F1) or "f1_pos" (reach F1)
    :param n_blocks: The number of contiguous time blocks per token (split_by="block" only)
    :param feature_window: None to feed the whole window, or the (first, last) second around the start the model may see
                           (fixed window only), e.g. (-2, 0) or (-3, -1); the events & the cleaning still use the whole
                           -pre..+fixed_after window, and the features keep the first pre points as the baseline
    :param clean_window: None to clean the events with the -pre..+fixed_after window, or the (first, last) second around
                         the start used by every overlap rule instead (fixed window only), e.g. (-3, -1)
    :param seed: The random seed

    :return: The per-token table
    """

    logger = SyncLogger(level="INFO")
    if selectivity != "none" and fixed_after is None:
        raise ValueError("the selectivity features need the fixed window")
    if mask_context != "none" and (fixed_after is None or feature_set != "extended"):
        raise ValueError("the context mask needs the fixed window & the extended features")
    if feature_window is not None and (fixed_after is None or mask_context != "none" or feature_window[1] > fixed_after
                                       or feature_window[1] - feature_window[0] < round(pre)):
        raise ValueError("feature_window needs the fixed window, no context mask, an end <= fixed_after and > pre points")
    if clean_window is not None and fixed_after is None:
        raise ValueError("clean_window needs the fixed window")
    # the windows are read from early enough for the feature window, the cleaning keeps using pre unless clean_window
    build_pre = pre if feature_window is None else max(pre, -feature_window[0])
    clean_pre, clean_after = (pre, fixed_after) if clean_window is None else (-clean_window[0], clean_window[1])
    sel_names = {"none": (), "basic": SELECTIVITY_FEATURES, "extended": EXTENDED_SELECTIVITY_FEATURES,
                 "decoders": DECODER_SELECTIVITY_FEATURES, "subtypes_glm": SUBTYPE_GLM_FEATURES,
                 "glm": EXTENDED_GLM_FEATURES, "glm_unmix": EXTENDED_UNMIX_FEATURES}[selectivity]
    window = "variable" if fixed_after is None else f"fixed{pre:g}+{fixed_after:g}"
    tag = f"{task}|nested|{feature_set}|{window}|{regularization}|drop={drop_overlap}|split={split_by}"
    if drop_prefixes:
        tag += f"|no={'+'.join(drop_prefixes)}" if len(drop_prefixes) <= 3 else f"|no={len(drop_prefixes)} prefixes"
    if selectivity != "none":
        tag += f"|selectivity={selectivity}"
    if mask_context != "none":
        tag += f"|mask={mask_context}"
    if weighting != "global":
        tag += f"|weight={weighting}"
    if cell_selection != "none":
        tag += f"|cells={cell_selection}"
    if circular_shift is not None:
        tag += f"|shift={circular_shift}"
    if stages is not None:
        tag += f"|stages={'+'.join(stages)}"
    if train_only_stages:
        tag += f"|train_only={'+'.join(train_only_stages)}"
    if select_by != "f1_macro":
        tag += f"|select={select_by}"
    if feature_window is not None:
        tag += f"|features={feature_window[0]:+g}..{feature_window[1]:+g}"
    if clean_window is not None:
        tag += f"|clean={clean_window[0]:+g}..{clean_window[1]:+g}"
    extract =event_features if feature_set == "basic" else extended_event_features
    feature_post = post if fixed_after is None else 0.0

    # build the cleaned events & features once, and the outer folds inside every token
    # the "fit" copies are used by the events that fit the features / templates / models, the others by the
    # validation & testing events; they only differ with mask_context="train"
    tables, features, windows, fit_features, fit_windows, n_dropped = [], [], [], [], [], 0
    offsets = {}
    for folder in src_folders:
        for token in fetch_tokens(folder=Path(folder), shard="all"):
            train_only = token.split("_")[-1] in train_only_stages
            if stages is not None and token.split("_")[-1] not in stages and not train_only:
                continue
            events, wins = build_spans(
                Path(folder) / f"{token}_B.csv", Path(folder) / f"{token}_S.csv",
                build_pre, post, max_len, transform=transform, fixed_after=fixed_after, circular_shift=circular_shift,
            )
            offsets[token] = events.attrs["shift"]
            y = events["behavior"].isin(TASKS[task]).to_numpy(dtype=int)
            # the context mask uses all the events of the token, before the cleaning
            masks = context_mask(events, pre, fixed_after) if mask_context != "none" else [None] * len(events)
            # a window is pure when no other event (of any behavior) falls inside it
            events = events.assign(pure_window=[bool(m.all()) for m in context_mask(events, pre, fixed_after)]) \
                if selectivity in ("subtypes_glm", "glm", "glm_unmix") else events
            if drop_overlap != "none":
                if drop_overlap == "symmetric":
                    # negatives whose window touches a reach, and reaches whose window touches a cont in vain
                    civ = (events["behavior"] == "cont in vain").to_numpy(dtype=int)
                    keep = (~((y == 0) & overlap_mask(events, y, "window_any", clean_pre, post, clean_after))
                            & ~((y == 1) & overlap_mask(events, civ, "window_any", clean_pre, post, clean_after)))
                else:
                    rule = drop_overlap.removesuffix("_dedup")
                    keep = ~(y == 0) | ~overlap_mask(events, y, rule, clean_pre, post, clean_after)
                    if drop_overlap.endswith("_dedup"):
                        keep &= first_negatives(events, y, keep, clean_pre, post, clean_after)
                n_dropped += int((~keep).sum())
                events = events.loc[keep].reset_index(drop=True)
                wins = [w for w, k in zip(wins, keep) if k]
                masks = [m for m, k in zip(masks, keep) if k]
                y = y[keep]
            if feature_window is not None:
                # the model only sees the points of the feature window, the events stay those of the whole window
                first, last = (int(round(build_pre + t)) for t in feature_window)
                wins = [w[:, first:last + 1] for w in wins]
            if split_by == "trial":
                groups = events["trial"].to_numpy()
            elif split_by == "block":
                # contiguous time blocks: the events sorted by start, cut into n_blocks runs of equal size
                rank = np.argsort(np.argsort(events["start"].to_numpy(), kind="stable"))
                groups = rank * n_blocks // len(events)
            else:
                groups = np.arange(len(events))
            raw_features = [extract(w, pre, feature_post) for w in wins]
            if mask_context != "none":
                masked_wins = [np.where(m[None, :], w, np.nan).astype(np.float32) for w, m in zip(wins, masks)]
                masked_features = [extended_event_features(w, pre, feature_post, m) for w, m in zip(masked_wins, masks)]
                events = events.copy()
                events["mask"] = ["".join("1" if v else "0" for v in m) for m in masks]
                events["n_masked"] = [int((~m).sum()) for m in masks]
            else:
                masked_wins, masked_features = wins, raw_features
            tables.append(events.assign(token=token, real=y, group=groups,
                                        outer=np.full(len(y), -1) if train_only else _assign_folds(y, groups, outer_folds, seed)))
            fit_features.extend(masked_features)
            fit_windows.extend(masked_wins)
            features.extend(masked_features if mask_context == "all" else raw_features)
            windows.extend(masked_wins if mask_context in ("all", "train+selectivity") else wins)

    table = pd.concat(tables, ignore_index=True)
    names = [n for n in feature_names(feature_set) if not n.startswith(tuple(drop_prefixes))]
    x = pd.DataFrame(np.stack(features), columns=feature_names(feature_set))[names]
    x_fit_all = pd.DataFrame(np.stack(fit_features), columns=feature_names(feature_set))[names]
    y = table["real"].to_numpy()
    tokens = table["token"].to_numpy()
    behaviors = table["behavior"].to_numpy()
    sel_names = window_feature_names(sel_names, pre, windows[0].shape[1])
    if feature_window is not None:
        # name the time-resolved selectivity by the real seconds of the feature window, e.g. -3..-1 instead of -2..0
        shift = int(round(feature_window[0] + pre))
        sel_names = tuple(f"sel_time_{int(n[9:]) + shift:+d}" if n.startswith("sel_time_") and n[9:].lstrip("+-").isdigit()
                          else n for n in sel_names)
    extra_ctx ={"starts": table["start"].to_numpy(), "pure": table["pure_window"].to_numpy(dtype=bool),
                 "subtypes": selectivity == "subtypes_glm", "unmix": selectivity == "glm_unmix"} \
        if selectivity in ("subtypes_glm", "glm", "glm_unmix") else None
    if extra_ctx is not None:
        logger.info(role="lgbm_nested",
                    message=f"[{tag}] pure reach windows={int((extra_ctx['pure'] & (y == 1)).sum())} "
                            f"mixed reach windows={int((~extra_ctx['pure'] & (y == 1)).sum())}")
    if mask_context != "none":
        masked = table.groupby("real")["n_masked"].agg(["mean", lambda s: (s > 0).mean()])
        logger.info(
            role="lgbm_nested",
            message=f"[{tag}] masked points per window (of {len(table['mask'].iloc[0])}): "
                    f"reach mean={masked.iloc[1, 0]:.2f} any={masked.iloc[1, 1]:.0%} | "
                    f"other mean={masked.iloc[0, 0]:.2f} any={masked.iloc[0, 1]:.0%}"
        )

    def design(fit_idx: np.ndarray, apply_idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_fit, x_apply = x_fit_all.iloc[fit_idx].reset_index(drop=True), x.iloc[apply_idx].reset_index(drop=True)
        if sel_names:
            s_fit, s_apply = selectivity_features(
                windows, tokens, y, fit_idx, apply_idx, pre, behaviors, selectivity in ("extended", "decoders", "subtypes_glm", "glm", "glm_unmix"),
                seed=seed, fit_windows=fit_windows, decoders=selectivity == "decoders", extra_ctx=extra_ctx,
            )
            x_fit = pd.concat([x_fit, pd.DataFrame(s_fit, columns=sel_names)], axis=1)
            x_apply = pd.concat([x_apply, pd.DataFrame(s_apply, columns=sel_names)], axis=1)
        return x_fit, x_apply

    def weights_of(idx: np.ndarray) -> np.ndarray | None:
        # "global": no sample weight (scale_pos_weight balances the classes); "session_balanced": every token the same
        # total weight, half reach / half non-reach inside it
        if weighting == "global":
            return None
        return sample_weights(tokens[idx], y[idx], weighting)

    names = names + list(sel_names)
    checkpoints =np.arange(tree_step, max_trees + 1, tree_step)
    logger.info(
        role="lgbm_nested",
        message=f"[{tag}] {table['token'].nunique()} tokens | {len(names)} features | dropped negatives={n_dropped} | "
                f"events={len(table)} reach={int(y.sum())} other={int((1 - y).sum())} | "
                f"{outer_folds} outer x {inner_folds} inner folds | trees <= {max_trees}"
    )
    if circular_shift is not None:
        pd.Series(offsets, name="shift_samples").rename_axis("token").to_csv(Path(tgt_folder) / "shifts.csv")
        logger.info(role="lgbm_nested", message=f"[{tag}] circular shift: signals rolled by "
                                                f"{min(offsets.values())}..{max(offsets.values())} samples")

    table["probability"] = np.nan
    table["prediction"] = -1
    fold_rows, importances = [], []
    all_windows, all_x = windows, x
    selection_rows, chosen_by_fold = [], {}
    if cell_selection != "none":
        # cell selection inside every token, from the outer training events of each fold only (the inner folds reuse
        # it); all folds are selected up front in one parallel pass, saved, and the workers released before training
        started = time.time()
        token_list = list(dict.fromkeys(tokens))
        outer = table["outer"].to_numpy()
        jobs = []
        for k in range(outer_folds):
            train_idx = np.flatnonzero(outer != k)
            for t in token_list:
                idx = train_idx[tokens[train_idx] == t]
                jobs.append((k, t, np.stack([all_windows[i] for i in idx]), y[idx]))
        selector = {"sps": sps_select, "sps_fixed": sps_fixed_select, "rfecv": rfecv_select}[cell_selection]
        picked = Parallel(n_jobs=min(16, os.cpu_count() or 1))(
            delayed(selector)(w, yy, pre=pre, seed=seed) for _, _, w, yy in jobs
        )
        get_reusable_executor().shutdown(wait=True)
        for (k, t, w, _), (sel, info) in zip(jobs, picked):
            chosen_by_fold.setdefault(k, {})[t] = sel
            row = {"outer_fold": k + 1, "token": t, "n_cells": w.shape[1], "n_selected": len(sel),
                   "cells": " ".join(map(str, sel))}
            if isinstance(info, dict):
                row["fallback"] = info["fallback"]
                row["auc_threshold"] = info["threshold"]
                row["visible_share"] = info["visible_share"]
                row["dz"] = " ".join(f"{v:.2f}" for v in info["dz"][sel])
            selection_rows.append(row)
        pd.DataFrame(selection_rows).to_csv(Path(tgt_folder) / "selected_cells.csv", index=False)
        logger.info(role="lgbm_nested", message=f"[{tag}] {cell_selection} cell selection of {outer_folds} folds "
                                                f"done in {time.time() - started:.0f} s, saved to selected_cells.csv")

    for k in range(outer_folds):
        train_mask = table["outer"].to_numpy() != k
        train_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)

        if cell_selection != "none":
            chosen = chosen_by_fold[k]
            token_list = list(chosen)
            jobs = [(t, all_windows[np.flatnonzero(tokens == t)[0]]) for t in token_list]
            windows = [w[chosen[t]] for w, t in zip(all_windows, tokens)]
            fit_windows = windows
            x = pd.DataFrame(np.stack([extract(w, pre, feature_post) for w in windows]),
                             columns=feature_names(feature_set))[[n for n in all_x.columns]]
            x_fit_all = x
            n_sel = np.array([len(chosen[t]) for t in token_list])
            n_all = np.array([w.shape[0] for _, w in jobs])
            logger.info(
                role="lgbm_nested",
                message=f"[{tag}] outer {k + 1}/{outer_folds} | {cell_selection} kept {n_sel.sum()}/{n_all.sum()} cells "
                        f"(per token median {np.median(n_sel):.0f}, min {n_sel.min()}, max {n_sel.max()})"
            )

        # inner folds inside every token of the outer training part
        inner = np.zeros(len(train_idx), dtype=int)
        train_table = table.iloc[train_idx]
        for token, part in train_table.groupby("token", sort=False):
            pos = np.flatnonzero(train_table["token"].to_numpy() == token)
            inner[pos] = _assign_folds(part["real"].to_numpy(), part["group"].to_numpy(), inner_folds, seed + 1)

        # out-of-fold probabilities of the outer training part for every candidate number of trees
        oof = np.zeros((len(train_idx), len(checkpoints)))
        for j in range(inner_folds):
            fit_idx, hold_idx = train_idx[inner != j], train_idx[inner == j]
            x_fit, x_hold = design(fit_idx, hold_idx)
            model = _make_model(max_trees, learning_rate, regularization, y[fit_idx], seed, weighting)
            model.fit(x_fit, y[fit_idx], sample_weight=weights_of(fit_idx))
            for c, n_trees in enumerate(checkpoints):
                oof[inner == j, c] = model.booster_.predict(x_hold, num_iteration=int(n_trees))
                if int(n_trees) % log_every == 0:
                    fit_prob = model.booster_.predict(x_fit, num_iteration=int(n_trees))
                    fit_f1 = f1_scores(y[fit_idx], (fit_prob >= 0.5).astype(int))
                    hold_f1 = f1_scores(y[hold_idx], (oof[inner == j, c] >= 0.5).astype(int))
                    logger.info(
                        role="lgbm_nested",
                        message=f"[{tag}] outer {k + 1}/{outer_folds} | inner {j + 1}/{inner_folds} | trees {int(n_trees):4d} | "
                                f"f1 reach/other/macro @0.5: fit={_f1_triplet(fit_f1)} holdout={_f1_triplet(hold_f1)} | "
                                f"AUC fit={roc_auc_score(y[fit_idx], fit_prob):.3f} "
                                f"holdout={roc_auc_score(y[hold_idx], oof[inner == j, c]):.3f}"
                    )
            last = oof[inner == j, -1]
            logger.info(
                role="lgbm_nested",
                message=f"[{tag}] outer {k + 1}/{outer_folds} | inner {j + 1}/{inner_folds} | "
                        f"fit={len(fit_idx)} holdout={len(hold_idx)} | holdout AUC@{max_trees}={roc_auc_score(y[hold_idx], last):.3f}"
            )

        # choose the number of trees & the threshold by the out-of-fold score (macro F1 or reach F1)
        curve = [_best_threshold_curve(y[train_idx], oof[:, c], select_by) for c in range(len(checkpoints))]
        best_c = int(np.argmax([score for _, score in curve]))
        n_trees, threshold = int(checkpoints[best_c]), curve[best_c][0]
        oof_pred = (oof[:, best_c] >= threshold).astype(int)
        oof_f1 = f1_scores(y[train_idx], oof_pred)

        # final model on the whole outer training part, evaluated once on the outer testing fold
        x_train, x_test = design(train_idx, test_idx)
        model = _make_model(n_trees, learning_rate, regularization, y[train_idx], seed, weighting)
        model.fit(x_train, y[train_idx], sample_weight=weights_of(train_idx))
        for n in [*range(log_every, n_trees, log_every), n_trees]:
            train_f1 = f1_scores(y[train_idx], (model.booster_.predict(x_train, num_iteration=n) >= threshold).astype(int))
            test_f1 = f1_scores(y[test_idx], (model.booster_.predict(x_test, num_iteration=n) >= threshold).astype(int))
            logger.info(
                role="lgbm_nested",
                message=f"[{tag}] outer {k + 1}/{outer_folds} | final | trees {n:4d}/{n_trees} | "
                        f"f1 reach/other/macro @{threshold:.2f}: train={_f1_triplet(train_f1)} test={_f1_triplet(test_f1)}"
            )
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= threshold).astype(int)
        table.loc[test_idx, "probability"] = prob
        table.loc[test_idx, "prediction"] = pred
        importances.append(model.booster_.feature_importance(importance_type="gain"))

        test_scores = _scores(y[test_idx], prob, pred)
        fold_rows.append({
            "outer_fold": k + 1, "n_train": len(train_idx), "n_test": len(test_idx),
            "test_reach": int(y[test_idx].sum()), "n_trees": n_trees, "threshold": threshold,
            "inner_oof_f1_reach": oof_f1["test_f1_pos"], "inner_oof_f1_macro": oof_f1["test_f1_macro"],
            **{f"test_{m}": v for m, v in test_scores.items()},
        })
        logger.info(
            role="lgbm_nested",
            message=f"[{tag}] outer {k + 1}/{outer_folds} | chosen trees={n_trees} threshold={threshold:.2f} | "
                    f"f1 reach/other/macro: inner-oof={_f1_triplet(oof_f1)} "
                    f"test={test_scores['f1_pos']:.3f}/{test_scores['f1_neg']:.3f}/{test_scores['f1_macro']:.3f} | "
                    f"test balanced={test_scores['balanced_accuracy']:.3f} AUC={test_scores['roc_auc']:.3f}"
        )

    # pooled over all tested events (each tested exactly once) & mean +- std over the outer folds;
    # the train-only tokens (outer fold -1) are left out of every score & output
    folds = pd.DataFrame(fold_rows)
    table = table.loc[table["outer"] >= 0].reset_index(drop=True)
    real, prob, pred = table["real"].to_numpy(), table["probability"].to_numpy(), table["prediction"].to_numpy()
    pooled = pd.DataFrame([{"scope": "pooled", **_scores(real, prob, pred)}])
    metric_cols = [c for c in folds.columns if c.startswith("test_") and c != "test_reach"]
    balanced = session_balanced_scores(real, prob, pred, table["token"].to_numpy())
    pooled = pd.concat([
        pooled,
        pd.DataFrame([{"scope": "session_balanced", **balanced}]),
        pd.DataFrame([{"scope": "fold_mean", **{c.replace("test_", ""): folds[c].mean() for c in metric_cols}}]),
        pd.DataFrame([{"scope": "fold_std", **{c.replace("test_", ""): folds[c].std() for c in metric_cols}}]),
    ], ignore_index=True)

    # per token over all its events
    rows = []
    for token, part in table.groupby("token", sort=False):
        tn, fp, fn, tp = confusion_matrix(part["real"], part["prediction"], labels=[0, 1]).ravel()
        s = _scores(part["real"].to_numpy(), part["probability"].to_numpy(), part["prediction"].to_numpy())
        mouse, fov, stage = token.split("_")
        rows.append({
            "token": token, "mouse": mouse, "FOV": fov, "stage": stage,
            "reach": int(part["real"].sum()), "other": int((part["real"] == 0).sum()),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "reach_F1": s["f1_pos"] if part["real"].any() else np.nan, "other_F1": s["f1_neg"],
            "balanced_accuracy": s["balanced_accuracy"], "roc_auc": s["roc_auc"],
        })
    per_token = pd.DataFrame(rows)

    logger.info(
        role="lgbm_nested",
        message=f"[{tag}] POOLED f1 reach/other/macro={pooled.loc[0, 'f1_pos']:.3f}/{pooled.loc[0, 'f1_neg']:.3f}/"
                f"{pooled.loc[0, 'f1_macro']:.3f} AUC={pooled.loc[0, 'roc_auc']:.3f} | SESSION-BALANCED ({balanced['tokens']} "
                f"tokens) f1 reach/other/macro={balanced['f1_pos']:.3f}/{balanced['f1_neg']:.3f}/{balanced['f1_macro']:.3f} "
                f"AUC={balanced['roc_auc']:.3f}"
    )
    importance = pd.DataFrame({"feature": names, "gain": np.mean(importances, axis=0)}).sort_values("gain", ascending=False)
    table.drop(columns=["group"]).to_csv(Path(tgt_folder) / "raw.csv", index=False)
    folds.to_csv(Path(tgt_folder) / "folds.csv", index=False)
    pooled.to_csv(Path(tgt_folder) / "pooled.csv", index=False)
    per_token.to_csv(Path(tgt_folder) / "per_token.csv", index=False)
    importance.to_csv(Path(tgt_folder) / "importance.csv", index=False)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        real, pred, display_labels=["Other", task], cmap="Reds", im_kw={"vmin": 0.0, "vmax": 1.0}, normalize="true", ax=ax,
    )
    plt.savefig(Path(tgt_folder) / "matrix.pdf")
    plt.close()

    logger.info(role="lgbm_nested", message=f"[{tag}] outputs saved to {tgt_folder}")
    return per_token
