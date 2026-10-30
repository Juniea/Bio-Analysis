import warnings

import numpy as np
from scipy.stats import wilcoxon

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# the five voters of tianyi's SPS-SELECT / RFECV-SELECT (tools.feature_engineer), with the classes balanced by weights
# instead of SMOTE so that no synthetic event ever reaches the cross-validation folds
SELECT_MODELS = ("LR", "SVM", "RF", "XGB", "LGB")


def _voter(name: str, y: np.ndarray, seed: int):
    pos_weight = (1 - y).sum() / max(y.sum(), 1)
    if name == "LR":
        return LogisticRegression(max_iter=50000, class_weight="balanced", random_state=seed)
    if name == "SVM":
        return LinearSVC(penalty="l1", max_iter=50000, class_weight="balanced", random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=1, random_state=seed)
    if name == "LGB":
        return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, num_leaves=31,
                                  class_weight="balanced", n_jobs=1, verbose=-1, random_state=seed)
    if name == "XGB":
        return xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                 scale_pos_weight=pos_weight, n_jobs=1, verbosity=0, random_state=seed)
    raise ValueError(f"unknown voter: {name}")


def _importance(estimator) -> np.ndarray:
    model = estimator.named_steps["model"]
    return np.abs(model.coef_).ravel() if hasattr(model, "coef_") else model.feature_importances_


def _folds(y: np.ndarray, cv: int, seed: int) -> StratifiedKFold | None:
    n_splits = min(cv, int(np.bincount(y, minlength=2).min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if n_splits >= 2 else None


def onset_response(
    wins: np.ndarray, y: np.ndarray, pre: float = 2.0, around: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    How every cell changes around the behavior start of the reaches: the mean of the `around` seconds from the start
    (t = 0 .. around - 1) minus the mean of the seconds before the start, per reach event

    :return: (effect size dz = mean / std of the per-event changes [n_cells], Wilcoxon signed-rank p-value [n_cells],
             consistency = share of the reaches changing in the cell's direction [n_cells]); dz > 0 brighter, < 0 darker
    """
    n_pre = int(round(pre))
    reach = np.nan_to_num(wins[y == 1])
    change = reach[:, :, n_pre:n_pre + around].mean(axis=2) - reach[:, :, :n_pre].mean(axis=2)   # [reaches, cells]
    with np.errstate(all="ignore"):
        dz = np.nan_to_num(change.mean(axis=0) / change.std(axis=0, ddof=1))
    p = np.ones(change.shape[1])
    for c in range(change.shape[1]):
        if len(change) >= 5 and np.any(change[:, c] != 0):
            p[c] = wilcoxon(change[:, c]).pvalue
    direction = np.where(dz >= 0, 1.0, -1.0)
    consistency = (np.sign(change) == direction).mean(axis=0) if len(change) else np.zeros(change.shape[1])
    return dz, p, consistency


def sps_select(
    wins: np.ndarray,
    y: np.ndarray,
    pre: float = 2.0,
    threshold: float = 0.55,
    standard: int = 4,
    response_p: float = 0.01,
    response_consistency: float = 0.7,
    visible_share: float = 0.7,
    threshold_step: float = 0.025,
    threshold_max: float = 0.9,
    score: str = "f1_macro",
    cv: int = 5,
    min_cells: int = 3,
    adaptive: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    SPS-SELECT on events: every cell alone (its whole window as the input) is cross-validated by the five voters,
    a voter votes for the cell when its `score` (macro F1 by default, any sklearn scorer name) reaches the threshold,
    a cell is kept with at least `standard` votes.
    The threshold starts at `threshold` and rises by threshold_step until at least visible_share of the kept cells
    visibly turn brighter or darker around the behavior start (see onset_response: p < response_p and at least
    response_consistency of the reaches changing in the same direction), never leaving fewer than min_cells;
    adaptive=False keeps the fixed `threshold` (tianyi's original rule, the visibility is then only reported)

    :param wins: [n_events, n_cells, n_points] the training events of one token
    :param y: [n_events] 1 = reach

    :return: (selected cell indexes, {"scores": [n_cells, len(SELECT_MODELS)], "dz", "p", "consistency": [n_cells],
             "threshold": the AUC threshold used, "visible_share": share of visible kept cells, "fallback": str});
             the min_cells best mean scores when even the first threshold keeps too few, all cells when a class is
             too rare to cross-validate
    """
    n_cells = wins.shape[1]
    folds = _folds(y, cv, seed)
    if folds is None:
        nan = np.full(n_cells, np.nan)
        return np.arange(n_cells), {"scores": np.full((n_cells, len(SELECT_MODELS)), np.nan), "dz": nan, "p": nan,
                                    "consistency": nan, "threshold": np.nan, "visible_share": np.nan,
                                    "fallback": "all cells (class too rare)"}
    scores = np.zeros((n_cells, len(SELECT_MODELS)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(n_cells):
            for m, name in enumerate(SELECT_MODELS):
                pipe = Pipeline([("scaler", StandardScaler()), ("model", _voter(name, y, seed))])
                scores[c, m] = cross_val_score(pipe, wins[:, c, :], y, cv=folds, scoring=score).mean()
    dz, p, consistency = onset_response(wins, y, pre)
    visible = (p < response_p) & (consistency >= response_consistency)
    scores = np.nan_to_num(scores)

    # raise the AUC threshold until the kept cells are mostly visible, keeping at least min_cells
    selected, used, share, fallback = None, np.nan, np.nan, ""
    for t in np.arange(threshold, threshold_max + 1e-9, threshold_step) if adaptive else [threshold]:
        kept = np.flatnonzero((scores >= t).sum(axis=1) >= standard)
        if len(kept) < min_cells:
            break
        selected, used, share = kept, float(t), float(visible[kept].mean())
        if share >= visible_share:
            break
    if selected is None:
        selected = np.sort(np.argsort(-scores.mean(axis=1))[:min(min_cells, n_cells)])
        share, fallback = float(visible[selected].mean()), "too few voted cells, best scores"
    elif adaptive and share < visible_share:
        fallback = f"visible share {share:.2f} at the highest threshold keeping {min_cells} cells"
    return selected, {"scores": scores, "dz": dz, "p": p, "consistency": consistency, "threshold": used,
                      "visible_share": share, "fallback": fallback}


def rfecv_select(
    wins: np.ndarray,
    y: np.ndarray,
    pre: float = 2.0,
    standard: int = 3,
    cv: int = 5,
    min_cells: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RFECV-SELECT on events: one feature per cell (its mean response after the start), every voter runs a recursive
    feature elimination with ROC-AUC cross-validation, a cell needing at least `standard` votes

    :param wins: [n_events, n_cells, n_points] the training events of one token
    :param y: [n_events] 1 = reach

    :return: (selected cell indexes, votes [n_cells, len(SELECT_MODELS)]); the cells with the most votes (then the best
             mean RFE ranks) when too few cells pass, all cells when a class is too rare to cross-validate
    """
    n_cells = wins.shape[1]
    folds = _folds(y, cv, seed)
    if folds is None:
        return np.arange(n_cells), np.full((n_cells, len(SELECT_MODELS)), np.nan)
    x = np.nan_to_num(wins[:, :, int(round(pre)):].mean(axis=2))
    votes, ranks = np.zeros((n_cells, len(SELECT_MODELS))), np.zeros((n_cells, len(SELECT_MODELS)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m, name in enumerate(SELECT_MODELS):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", _voter(name, y, seed))])
            selector = RFECV(pipe, step=1, cv=folds, scoring="roc_auc", importance_getter=_importance).fit(x, y)
            votes[:, m], ranks[:, m] = selector.support_, selector.ranking_
    hits = votes.sum(axis=1)
    selected = np.flatnonzero(hits >= standard)
    if len(selected) < min_cells:
        order = np.lexsort((ranks.mean(axis=1), -hits))
        selected = np.sort(order[:min(min_cells, n_cells)])
    return selected, votes


def sps_fixed_select(wins: np.ndarray, y: np.ndarray, pre: float = 2.0, seed: int = 42) -> tuple[np.ndarray, dict]:
    """SPS-SELECT with the fixed macro F1 threshold 0.6 (no raising, no visibility rule), at least 4 votes"""
    return sps_select(wins, y, pre=pre, threshold=0.6, adaptive=False, seed=seed)


__all__ = ["SELECT_MODELS", "onset_response", "sps_select", "sps_fixed_select", "rfecv_select"]
