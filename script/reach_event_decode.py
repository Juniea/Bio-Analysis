import sys
import zlib
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_event_clean import build_events, src_folders

warnings.filterwarnings("ignore")

OFFSETS     = np.arange(-2, 5)          # sample points relative to event start (s)
PRE, EVENT  = slice(0, 2), slice(2, 7)  # points -2..-1 / 0..+4
POST        = 6                         # point +4
TREE_GRID   = np.arange(10, 1001, 10)
THR_GRID    = np.linspace(0.05, 0.95, 91)
PCA_GRID    = (10, 20, 50, 100)                 # PCA components for the pca_* models
LR_C_GRID   = (0.001, 0.01, 0.1, 1.0)           # inverse regularization for pca_lr
CONFUSERS   = {
    "both_in_vain": {"both in vain"},
    "walk": {"walk"},
    "noise": {"noise generation", "noise genearation", "noist generation"},
    "cont_in_vain": {"cont in vain"},
}
LGB_PARAMS  = dict(
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=80,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_lambda=10.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# ---------------------------------------------------------------- 5. input windows
def remove_global(x: np.ndarray, mode: str) -> np.ndarray:
    """
    remove the FOV-wide shared component from a (frames, cells) signal over the whole session (label-free),
    then z-score every cell again

    :param mode: "mean" = regress every cell on the population mean; "pc1" = subtract the first principal component
    """
    xc = x - x.mean(axis=0)
    if mode == "mean":
        g = xc.mean(axis=1)
        beta = (xc.T @ g) / max(g @ g, 1e-12)
        xc = xc - np.outer(g, beta)
    elif mode == "pc1":
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        xc = xc - s[0] * np.outer(u[:, 0], vt[0])
    else:
        raise ValueError(f"unknown global mode: {mode}")
    return (xc - xc.mean(axis=0)) / (xc.std(axis=0) + 1e-6)


def load_windows(
        events: pd.DataFrame,
        norm: str = "zscore",
        glob: str = "none",
        skew: bool = False,
) -> dict[str, np.ndarray]:
    """
    per session: normalize every cell over the whole session, then interpolate at start + (-2..+4) s

    :param norm: "zscore" = (x - mean) / std; "zero_anchored" = x / std, so silence stays at 0 for every cell
                 (with ~60 % zeros, z-scoring maps silence to a cell-specific -mean/std that fingerprints the session)
    :param glob: "none" | "mean" | "pc1", see remove_global (applied after `norm`)
    :param skew: negative control, permute every cell's raw frames on the time axis independently BEFORE the
                 interpolation of the event windows, so the signal has no relation to behavior

    :return: {token: array (n_events_in_session, n_cells, 7)} in the row order of `events`
    """
    paths = {p.stem[:-2]: p for folder in src_folders for p in folder.glob("*_S.csv")}
    windows = {}
    for token, ev in events.groupby("token", sort=False):
        sig = pd.read_csv(paths[token])
        mouse = token.split("_")[0]
        sig_fps = 1 / 1.000045 if mouse == "K215" else 1 / 1.001286
        sig_time = sig.pop("x0000").to_numpy(dtype=float) / sig_fps
        x = sig.to_numpy(dtype=float)
        if norm == "zscore":
            x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        elif norm == "zero_anchored":
            x = x / (x.std(axis=0) + 1e-6)
        else:
            raise ValueError(f"unknown norm: {norm}")
        if glob != "none":
            x = remove_global(x, glob)
        if skew:
            rng = np.random.default_rng(zlib.crc32(token.encode()))
            x = np.stack([rng.permutation(x[:, c]) for c in range(x.shape[1])], axis=1)

        t = ev["start"].to_numpy()[:, None] + OFFSETS[None, :]
        w = np.stack([np.interp(t, sig_time, x[:, c]) for c in range(x.shape[1])], axis=1)
        windows[token] = w
    return windows


# ---------------------------------------------------------------- 6a. handcrafted features
STAT_NAMES = [
    "mean", "median", "max", "min", "std", "range", "q25", "q75", "q90", "skew", "kurtosis", "energy",
    "auc", "slope", "first_last", "peak_time", "onset_time", "mean_abs_diff", "max_abs_diff", "autocorr1", "n_peaks",
    "pre_mean", "pre_max", "pre_std", "event_mean", "event_max", "event_std", "post_mean", "post_max",
    "event_minus_pre", "post_minus_event",
]
AGG_NAMES = ["mean", "std", "max", "min", "median", "q10", "q25", "q75", "q90", "top5"]
POP_NAMES = [
    "pop_mean", "pop_max", "pop_std", "pop_slope", "pop_event_minus_pre", "synchrony",
    "frac_event_minus_pre_gt_0.5", "frac_event_minus_pre_gt_1.0", "frac_active",
]


def _slope(x: np.ndarray) -> np.ndarray:
    tc = OFFSETS - OFFSETS.mean()
    return (x * tc).sum(axis=-1) / (tc ** 2).sum()


def cell_stats(w: np.ndarray) -> np.ndarray:
    """
    :param w: (n, cells, 7)
    :return: (n, cells, 31)
    """
    mean, mx, mn = w.mean(-1), w.max(-1), w.min(-1)
    q25, q75, q90 = np.quantile(w, [0.25, 0.75, 0.9], axis=-1)
    d = np.diff(w, axis=-1)

    above = w > 1
    onset = np.where(above.any(-1), above.argmax(-1) / 6, 1.0)

    xc = w - mean[..., None]
    den = (xc ** 2).sum(-1)
    autocorr = np.where(den > 0, (xc[..., :-1] * xc[..., 1:]).sum(-1) / np.where(den > 0, den, 1), 0.0)

    mid = w[..., 1:-1]
    n_peaks = ((mid > w[..., :-2]) & (mid > w[..., 2:]) & (mid > 1)).sum(-1)

    pre, ev, post = w[..., PRE], w[..., EVENT], w[..., POST]
    stats = [
        mean, np.median(w, -1), mx, mn, w.std(-1), mx - mn, q25, q75, q90,
        skew(w, axis=-1), kurtosis(w, axis=-1), (w ** 2).mean(-1),
        mean * 7, _slope(w), w[..., -1] - w[..., 0], w.argmax(-1) / 6, onset,
        np.abs(d).mean(-1), np.abs(d).max(-1), autocorr, n_peaks,
        pre.mean(-1), pre.max(-1), pre.std(-1), ev.mean(-1), ev.max(-1), ev.std(-1), post, post,
        ev.mean(-1) - pre.mean(-1), post - ev.mean(-1),
    ]
    return np.stack(stats, axis=-1).astype(float)


def aggregate_cells(s: np.ndarray) -> np.ndarray:
    """
    :param s: (n, cells, k)
    :return: (n, k, 10)
    """
    filled = np.where(np.isnan(s), -np.inf, s)
    top = np.sort(filled, axis=1)[:, -5:, :]
    top = np.where(np.isinf(top), np.nan, top)
    aggs = [
        np.nanmean(s, 1), np.nanstd(s, 1), np.nanmax(s, 1), np.nanmin(s, 1), np.nanmedian(s, 1),
        *np.nanquantile(s, [0.1, 0.25, 0.75, 0.9], axis=1), np.nanmean(top, 1),
    ]
    return np.stack(aggs, axis=-1)


def population_stats(w: np.ndarray) -> np.ndarray:
    pop = w.mean(1)
    sd = w.std(-1, keepdims=True)
    z = np.where(sd > 0, (w - w.mean(-1, keepdims=True)) / np.where(sd > 0, sd, 1), np.nan)
    corr = np.einsum("nct,ndt->ncd", np.nan_to_num(z), np.nan_to_num(z)) / w.shape[-1]
    valid = ~np.isnan(z[..., 0])
    pair = valid[:, :, None] & valid[:, None, :] & ~np.eye(w.shape[1], dtype=bool)[None]
    synchrony = np.where(pair.sum((1, 2)) > 0, (corr * pair).sum((1, 2)) / np.maximum(pair.sum((1, 2)), 1), np.nan)
    emp = w[..., EVENT].mean(-1) - w[..., PRE].mean(-1)
    return np.stack([
        pop.mean(-1), pop.max(-1), pop.std(-1), _slope(pop), pop[:, EVENT].mean(-1) - pop[:, PRE].mean(-1),
        synchrony, (emp > 0.5).mean(1), (emp > 1.0).mean(1), (w.max(-1) > w.min(-1)).mean(1),
    ], axis=-1)


def handcrafted(w: np.ndarray) -> pd.DataFrame:
    agg = aggregate_cells(cell_stats(w)).reshape(len(w), -1)
    names = [f"{s}_{a}" for s in STAT_NAMES for a in AGG_NAMES]
    df = pd.DataFrame(np.concatenate([agg, population_stats(w)], axis=1), columns=names + POP_NAMES)
    return df.loc[:, [c for c in df.columns if not c.startswith("min_")]]


# ---------------------------------------------------------------- 6b. reach-selectivity features
SEL_NAMES = (
    ["sel_weighted", "sel_weighted_delta", "sel_lda", "sel_corr_diff", "sel_dist_diff",
     "sel_top_reach", "sel_top_other", "sel_top_diff"]
    + [f"sel_time_{o:+d}" for o in OFFSETS]
    + [f"sel_vs_{k}{s}" for k in CONFUSERS for s in ("", "_delta")]
    + ["sel_logreg", "sel_shrink_lda"]
)


def _dprime(v1: np.ndarray, v0: np.ndarray) -> dict:
    mu1, mu0 = v1.mean(0), v0.mean(0)
    ss = ((v1 - mu1) ** 2).sum(0) + ((v0 - mu0) ** 2).sum(0)
    sd = np.sqrt(ss / (len(v1) + len(v0) - 2) + 0.01)
    return {"mu1": mu1, "mu0": mu0, "sd": sd, "d": (mu1 - mu0) / sd}


def _weighted(v: np.ndarray, t: dict) -> np.ndarray:
    norm = np.abs(t["d"]).sum()
    if norm == 0:
        return np.full(len(v), np.nan)
    return (((v - (t["mu1"] + t["mu0"]) / 2) / t["sd"]) * t["d"]).sum(1) / norm


def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum())
    return np.where(den > 0, (a @ b) / np.where(den > 0, den, 1), np.nan)


class Template:
    """
    reach-selectivity template of one session, learned from labelled events only
    """

    def __init__(self, w: np.ndarray, y: np.ndarray, beh: np.ndarray):
        self.ok = (y == 1).sum() >= 2 and (y == 0).sum() >= 2
        if not self.ok:
            return

        resp, delta, flat = self._views(w)
        r, o = y == 1, y == 0
        self.resp = _dprime(resp[r], resp[o])
        self.delta = _dprime(delta[r], delta[o])
        self.time = [_dprime(w[r, :, i], w[o, :, i]) for i in range(len(OFFSETS))]

        self.vs = {}
        for name, labels in CONFUSERS.items():
            m = np.isin(beh, list(labels))
            self.vs[name] = (_dprime(resp[r], resp[m]), _dprime(delta[r], delta[m])) if m.sum() >= 3 else None

        lda = _dprime(flat[r], flat[o])
        self.lda_w = (lda["mu1"] - lda["mu0"]) / lda["sd"] ** 2
        proj = flat @ self.lda_w
        self.lda_m, self.lda_s = proj.mean(), proj.std()

        k = max(1, round(0.1 * w.shape[1]))
        order = np.argsort(self.resp["d"])
        self.top_reach, self.top_other = order[-k:], order[:k]

        self.logreg = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=500)
        ).fit(flat, y)
        self.shrink = make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
        ).fit(flat, y)

    @staticmethod
    def _views(w: np.ndarray):
        resp = w[..., EVENT].mean(-1)
        return resp, resp - w[..., PRE].mean(-1), w.reshape(len(w), -1)

    def transform(self, w: np.ndarray) -> np.ndarray:
        if not self.ok:
            return np.full((len(w), len(SEL_NAMES)), np.nan)

        resp, delta, flat = self._views(w)
        t = self.resp
        zr = (resp - (t["mu1"] + t["mu0"]) / 2) / t["sd"]
        top_r, top_o = zr[:, self.top_reach].mean(1), zr[:, self.top_other].mean(1)
        dist = (np.linalg.norm((resp - t["mu0"]) / t["sd"], axis=1)
                - np.linalg.norm((resp - t["mu1"]) / t["sd"], axis=1)) / np.sqrt(w.shape[1])

        cols = [
            _weighted(resp, t), _weighted(delta, self.delta),
            (flat @ self.lda_w - self.lda_m) / (self.lda_s + 1e-12),
            _rowcorr(resp, t["mu1"]) - _rowcorr(resp, t["mu0"]),
            dist, top_r, top_o, top_r - top_o,
        ]
        cols += [_weighted(w[:, :, i], tt) for i, tt in enumerate(self.time)]
        for name in CONFUSERS:
            pair = self.vs[name]
            cols += [np.full(len(w), np.nan)] * 2 if pair is None else [_weighted(resp, pair[0]), _weighted(delta, pair[1])]
        cols += [self.logreg.decision_function(flat), self.shrink.decision_function(flat)]
        return np.stack(cols, axis=1)


# ---------------------------------------------------------------- 6c. general (cross-session) shape features
# (the population-mean projection equals gen_shape_mean, since the projection is linear)
GEN_NAMES = ["gen_shape_mean", "gen_shape_top", "gen_shape_q90", "gen_shape_frac_pos", "gen_shape_shrunk"]
GEN_SHRINK_N = 10   # lambda = n_reach / (n_reach + GEN_SHRINK_N)


def general_shape(sessions: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """
    cell-identity-free reach response shape: in every session take the top 10 % reach-selective cells (d' on resp),
    average their (reach mean - non-reach mean) 7-point curves, then average over sessions weighted by #reach

    :param sessions: [(windows (n, cells, 7), labels (n,)), ...] of labelled events
    :return: centered unit-norm (7,) shape, or None when no session has >= 2 events of each class
    """
    curves, weights = [], []
    for w, y in sessions:
        r, o = y == 1, y == 0
        if r.sum() < 2 or o.sum() < 2:
            continue
        resp = w[..., EVENT].mean(-1)
        d = _dprime(resp[r], resp[o])["d"]
        top = np.argsort(d)[-max(1, round(0.1 * w.shape[1])):]
        curves.append((w[r].mean(0) - w[o].mean(0))[top].mean(0))
        weights.append(r.sum())
    if not curves:
        return None
    g = np.average(np.stack(curves), axis=0, weights=weights)
    g = g - g.mean()
    norm = np.linalg.norm(g)
    return g / norm if norm > 0 else None


def general_features(w: np.ndarray, shape: np.ndarray | None, w_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """
    :param w: (n, cells, 7) events to describe
    :param shape: general shape from `general_shape`
    :param w_fit, y_fit: this session's labelled events, only used for the shrunk cell weights
    :return: (n, len(GEN_NAMES))
    """
    if shape is None:
        return np.full((len(w), len(GEN_NAMES)), np.nan)

    proj = ((w - w.mean(-1, keepdims=True)) * shape).sum(-1)    # (n, cells): how much each cell follows the shape
    k = max(1, round(0.1 * w.shape[1]))

    # shrink this session's d' cell weights toward uniform weights when it has few reach events
    uniform = np.ones(w.shape[1])
    n_reach = int((y_fit == 1).sum())
    if n_reach >= 2 and (y_fit == 0).sum() >= 2:
        resp = w_fit[..., EVENT].mean(-1)
        d = _dprime(resp[y_fit == 1], resp[y_fit == 0])["d"]
        d = d / max(np.abs(d).mean(), 1e-12)
        lam = n_reach / (n_reach + GEN_SHRINK_N)
        weight = lam * d + (1 - lam) * uniform
    else:
        weight = uniform

    return np.stack([
        proj.mean(1),
        np.sort(proj, axis=1)[:, -k:].mean(1),
        np.quantile(proj, 0.9, axis=1),
        (proj > 0.5).mean(1),
        (proj * weight).sum(1) / np.abs(weight).sum(),
    ], axis=1)


# ---------------------------------------------------------------- label control: segment shuffle
def segment_shuffle_labels(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    negative control on the labels: per session, order the events by start time, split the label sequence into
    runs of consecutive 1s / 0s, shuffle the 1-runs among themselves and the 0-runs among themselves (alternation,
    starting value and every run length kept), and hand the new sequence back to the events in time order.
    Event times, signal and behavior names are untouched.
    """
    out = []
    for token, g in events.groupby("token", sort=False):
        g = g.sort_values("start").copy()
        y = g["label"].to_numpy()
        change = np.flatnonzero(np.diff(y)) + 1
        runs = [(int(y[a]), int(b - a)) for a, b in zip(np.r_[0, change], np.r_[change, len(y)])]

        rng = np.random.default_rng(zlib.crc32(token.encode()) + seed)
        ones = [n for v, n in runs if v == 1]
        zeros = [n for v, n in runs if v == 0]
        rng.shuffle(ones)
        rng.shuffle(zeros)
        ones, zeros = iter(ones), iter(zeros)
        g["label"] = np.concatenate([np.full(next(ones) if v else next(zeros), v) for v, _ in runs])
        out.append(g)
    return pd.concat(out).sort_index()


# ---------------------------------------------------------------- 7. folds
def make_folds(y: np.ndarray, seed: int) -> np.ndarray:
    """
    per-session fold ids; every event is its own group
    """
    n = len(y)
    fold = np.zeros(n, dtype=int)
    groups = np.arange(n)
    if min((y == 1).sum(), (y == 0).sum()) >= 5 and n >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), y, groups)
    elif n >= 2:
        splitter = GroupKFold(n_splits=min(5, n)).split(np.zeros(n), y, groups)
    else:
        return fold
    for k, (_, te) in enumerate(splitter):
        fold[te] = k
    return fold


class Decoder:
    def __init__(
            self,
            events: pd.DataFrame,
            extras: tuple[str, ...],
            norm: str = "zscore",
            glob: str = "none",
            general: bool = False,
            model: str = "lgb",
            skew: bool = False,
            blocked: int = 0,
    ):
        self.general = general
        self.model = model
        self.blocked = blocked
        self.ev = events.reset_index(drop=True)
        self.y = self.ev["label"].to_numpy()
        self.beh = self.ev["behavior"].to_numpy()
        self.sess = self.ev["token"].to_numpy()

        windows = load_windows(self.ev, norm=norm, glob=glob, skew=skew)
        self.w = windows
        self.pos = self.ev.groupby("token").cumcount().to_numpy()
        self.hand = pd.concat(
            [handcrafted(windows[tok]).set_index(self.ev.index[self.sess == tok]) for tok in windows]
        ).sort_index()

        extra = [pd.get_dummies(self.ev[c], prefix=c, dtype=float) for c in extras]
        self.extra = pd.concat(extra, axis=1) if extra else pd.DataFrame(index=self.ev.index)

    def _w(self, idx: np.ndarray) -> np.ndarray:
        return self.w[self.sess[idx[0]]][self.pos[idx]]

    def sel_apply(self, fit_idx: np.ndarray, apply_idx: np.ndarray) -> np.ndarray:
        out = np.full((len(apply_idx), len(SEL_NAMES)), np.nan)
        for tok in np.unique(self.sess[apply_idx]):
            f = fit_idx[self.sess[fit_idx] == tok]
            a = self.sess[apply_idx] == tok
            if len(f) == 0:
                continue
            tpl = Template(self._w(f), self.y[f], self.beh[f])
            out[a] = tpl.transform(self._w(apply_idx[a]))
        return out

    def sel_crossfit(self, idx: np.ndarray, seed: int = 42) -> np.ndarray:
        out = np.full((len(idx), len(SEL_NAMES)), np.nan)
        for tok in np.unique(self.sess[idx]):
            m = np.flatnonzero(self.sess[idx] == tok)
            fold = make_folds(self.y[idx[m]], seed)
            for k in np.unique(fold):
                out[m[fold == k]] = self.sel_apply(idx[m[fold != k]], idx[m[fold == k]])
        return out

    def gen_apply(self, fit_idx: np.ndarray, apply_idx: np.ndarray) -> np.ndarray:
        """
        general shape learned from the labelled `fit_idx` events of ALL sessions, applied to `apply_idx`
        """
        fit_sessions = [
            (self._w(fit_idx[self.sess[fit_idx] == tok]), self.y[fit_idx[self.sess[fit_idx] == tok]])
            for tok in np.unique(self.sess[fit_idx])
        ]
        shape = general_shape(fit_sessions)

        out = np.full((len(apply_idx), len(GEN_NAMES)), np.nan)
        for tok in np.unique(self.sess[apply_idx]):
            a = self.sess[apply_idx] == tok
            f = fit_idx[self.sess[fit_idx] == tok]
            w_fit = self._w(f) if len(f) else np.zeros((0,) + self.w[tok].shape[1:])
            out[a] = general_features(self._w(apply_idx[a]), shape, w_fit, self.y[f])
        return out

    def gen_crossfit(self, idx: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        cross-fitting for the events that train LightGBM: per-session 5 folds, fold k across all sessions
        gets the general shape learned from the other folds of all sessions
        """
        fold = self.session_folds(idx, seed=seed)
        out = np.full((len(idx), len(GEN_NAMES)), np.nan)
        for k in np.unique(fold):
            out[fold == k] = self.gen_apply(idx[fold != k], idx[fold == k])
        return out

    def matrix(self, idx: np.ndarray, sel: np.ndarray, gen: np.ndarray | None = None) -> pd.DataFrame:
        parts = [
            self.hand.iloc[idx].reset_index(drop=True),
            pd.DataFrame(sel, columns=SEL_NAMES),
            self.extra.iloc[idx].reset_index(drop=True),
        ]
        if gen is not None:
            parts.append(pd.DataFrame(gen, columns=GEN_NAMES))
        return pd.concat(parts, axis=1)

    def train_matrix(self, idx: np.ndarray) -> pd.DataFrame:
        gen = self.gen_crossfit(idx) if self.general else None
        return self.matrix(idx, self.sel_crossfit(idx), gen)

    def apply_matrix(self, fit_idx: np.ndarray, apply_idx: np.ndarray) -> pd.DataFrame:
        gen = self.gen_apply(fit_idx, apply_idx) if self.general else None
        return self.matrix(apply_idx, self.sel_apply(fit_idx, apply_idx), gen)

    def _model(self, n_estimators: int, y_fit: np.ndarray) -> lgb.LGBMClassifier:
        spw = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
        return lgb.LGBMClassifier(n_estimators=n_estimators, scale_pos_weight=spw, **LGB_PARAMS)

    def session_folds(self, idx: np.ndarray, seed: int) -> np.ndarray:
        fold = np.zeros(len(idx), dtype=int)
        for tok in np.unique(self.sess[idx]):
            m = self.sess[idx] == tok
            fold[m] = make_folds(self.y[idx[m]], seed)
        return fold

    def configs(self) -> list[tuple]:
        """
        every hyper-parameter combination the inner loop chooses from (the threshold is chosen on top)
        """
        if self.model == "lgb":
            return [("lgb", int(n)) for n in TREE_GRID]
        if self.model == "pca_lr":
            return [("pca_lr", k, c) for k in PCA_GRID for c in LR_C_GRID]
        if self.model == "pca_lgb":
            return [("pca_lgb", k, int(n)) for k in PCA_GRID for n in TREE_GRID]
        raise ValueError(f"unknown model: {self.model}")

    @staticmethod
    def _pca(k: int) -> Pipeline:
        # median imputation (selectivity features can be NaN) -> z-score -> PCA, all fitted on the training part only
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=k, random_state=42)),
        ])

    def predict_configs(self, x_fit: pd.DataFrame, y_fit: np.ndarray, x_val: pd.DataFrame, configs: list[tuple]) -> np.ndarray:
        """
        fit every config on (x_fit, y_fit) and return the validation probabilities, (len(x_val), len(configs));
        the tree-count axis shares a single 1000-tree fit, as in the original LightGBM loop
        """
        out = np.zeros((len(x_val), len(configs)))
        col = {cfg: i for i, cfg in enumerate(configs)}
        if self.model == "lgb":
            model = self._model(TREE_GRID[-1], y_fit).fit(x_fit, y_fit)
            for n in TREE_GRID:
                out[:, col[("lgb", int(n))]] = model.predict_proba(x_val, num_iteration=n)[:, 1]
            return out

        for k in PCA_GRID:
            prep = self._pca(k).fit(x_fit)
            z_fit, z_val = prep.transform(x_fit), prep.transform(x_val)
            if self.model == "pca_lr":
                for c in LR_C_GRID:
                    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(z_fit, y_fit)
                    out[:, col[("pca_lr", k, c)]] = lr.predict_proba(z_val)[:, 1]
            else:
                model = self._model(TREE_GRID[-1], y_fit).fit(z_fit, y_fit)
                for n in TREE_GRID:
                    out[:, col[("pca_lgb", k, int(n))]] = model.predict_proba(z_val, num_iteration=n)[:, 1]
        return out

    def fit_predict(self, cfg: tuple, x_fit: pd.DataFrame, y_fit: np.ndarray, x_te: pd.DataFrame) -> np.ndarray:
        if cfg[0] == "lgb":
            return self._model(cfg[1], y_fit).fit(x_fit, y_fit).predict_proba(x_te)[:, 1]
        prep = self._pca(cfg[1]).fit(x_fit)
        z_fit, z_te = prep.transform(x_fit), prep.transform(x_te)
        if cfg[0] == "pca_lr":
            lr = LogisticRegression(C=cfg[2], class_weight="balanced", max_iter=2000).fit(z_fit, y_fit)
            return lr.predict_proba(z_te)[:, 1]
        return self._model(cfg[2], y_fit).fit(z_fit, y_fit).predict_proba(z_te)[:, 1]

    def blocked_folds(self, k: int) -> np.ndarray:
        """
        per session: order the events by start time and cut them into k contiguous blocks of (nearly) equal size
        """
        fold = np.zeros(len(self.ev), dtype=int)
        for tok in np.unique(self.sess):
            idx = np.flatnonzero(self.sess == tok)
            order = idx[np.argsort(self.ev["start"].to_numpy()[idx], kind="stable")]
            fold[order] = np.arange(len(order)) * k // len(order)
        return fold

    def run(self, folds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
        all_idx = np.arange(len(self.ev))
        outer = self.blocked_folds(self.blocked) if self.blocked else self.session_folds(all_idx, seed=42)
        proba = np.full(len(all_idx), np.nan)
        pred = np.zeros(len(all_idx), dtype=int)
        configs = self.configs()
        self.choices = []

        for k in folds:
            tr, te = all_idx[outer != k], all_idx[outer == k]
            inner = self.session_folds(tr, seed=43)

            # inner out-of-fold probabilities over every config
            oof = np.zeros((len(tr), len(configs)))
            for j in range(5):
                fit, val = tr[inner != j], tr[inner == j]
                if len(val) == 0:
                    continue
                oof[inner == j] = self.predict_configs(
                    self.train_matrix(fit), self.y[fit], self.apply_matrix(fit, val), configs
                )

            # pick (config, threshold) by inner OOF macro F1
            f1 = macro_f1_grid(self.y[tr], oof, THR_GRID)
            i, t = np.unravel_index(np.argmax(f1), f1.shape)
            cfg_best, thr_best = configs[i], float(THR_GRID[t])

            proba[te] = self.fit_predict(cfg_best, self.train_matrix(tr), self.y[tr], self.apply_matrix(tr, te))
            pred[te] = (proba[te] >= thr_best).astype(int)

            inner_pred = (oof[:, i] >= thr_best).astype(int)
            n_best = "/".join(str(v) for v in cfg_best[1:])
            c = {"fold": k + 1, "trees": n_best, "threshold": round(thr_best, 3),
                 "inner_macro_f1": float(f1[i, t]),
                 "inner_reach_f1": f1_score(self.y[tr], inner_pred),
                 "test_reach_f1": f1_score(self.y[te], pred[te]),
                 "test_macro_f1": f1_score(self.y[te], pred[te], average="macro"),
                 "test_auc": roc_auc_score(self.y[te], proba[te])}
            self.choices.append(c)
            print(f"  outer fold {k + 1}: {cfg_best[0]}={n_best} thr={thr_best:.2f} "
                  f"inner reach F1={c['inner_reach_f1']:.3f} macro={c['inner_macro_f1']:.3f} | "
                  f"test reach F1={c['test_reach_f1']:.3f} macro={c['test_macro_f1']:.3f} AUC={c['test_auc']:.3f}",
                  flush=True)

        done = np.isin(outer, folds)
        return self.ev.assign(proba=proba, pred=pred, outer_fold=outer + 1).loc[done]


def macro_f1_grid(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """
    :return: (len(trees), len(thr)) macro F1, prediction = p >= thr
    """
    hat = p[:, :, None] >= thr[None, None, :]
    pos = (y == 1)[:, None, None]
    tp = (hat & pos).sum(0)
    fp = (hat & ~pos).sum(0)
    fn = (~hat & pos).sum(0)
    tn = (~hat & ~pos).sum(0)
    f1_1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_0 = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return (f1_1 + f1_0) / 2


# ---------------------------------------------------------------- 9. evaluation
def scores(df: pd.DataFrame) -> dict:
    y, p, s = df["label"].to_numpy(), df["pred"].to_numpy(), df["proba"].to_numpy()
    out = {"n": len(y), "reach": int(y.sum()),
           "nonreach_f1": f1_score(y, p, pos_label=0, zero_division=0)}
    if 0 < y.sum() < len(y):
        out |= {"reach_f1": f1_score(y, p, pos_label=1, zero_division=0),
                "bal_acc": balanced_accuracy_score(y, p),
                "auc": roc_auc_score(y, s)}
        out["macro_f1"] = (out["reach_f1"] + out["nonreach_f1"]) / 2
    return out


def report(res: pd.DataFrame) -> dict:
    cols = ["reach_f1", "nonreach_f1", "macro_f1", "bal_acc", "auc"]
    pooled = scores(res)
    per_sess = pd.DataFrame({tok: scores(g) for tok, g in res.groupby("token")}).T
    per_stage = pd.DataFrame({st: scores(g) for st, g in res.groupby("stage")}).T

    print("  pooled      :", "  ".join(f"{c}={pooled[c]:.3f}" for c in cols))
    print("  session mean:", "  ".join(f"{c}={per_sess[c].astype(float).mean():.3f}" for c in cols),
          f"(single-class sessions: {per_sess['reach_f1'].isna().sum()})")
    print("  per stage (pooled within stage):")
    print(per_stage.loc[:, ["n", "reach"] + cols].astype(float).round(3).to_string())
    return {"pooled": pooled, "session_mean": per_sess[cols].astype(float).mean().to_dict()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=None, help="e.g. D1 D2 D7 D8 M4 M5; omit for all")
    parser.add_argument("--extras", nargs="*", default=[], help="extra one-hot columns, e.g. fov stage")
    parser.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="outer folds to run (1-based)")
    parser.add_argument("--norm", default="zscore", choices=["zscore", "zero_anchored"], help="per-cell signal normalization")
    parser.add_argument("--global", dest="glob", default="none", choices=["none", "mean", "pc1"],
                        help="remove the FOV-wide shared component per session")
    parser.add_argument("--general", action="store_true", help="add the cross-session general shape features")
    parser.add_argument("--model", default="lgb", choices=["lgb", "pca_lr", "pca_lgb"],
                        help="lgb = regularized LightGBM on all features; pca_* = impute/scale/PCA then LR or LightGBM")
    parser.add_argument("--skew", action="store_true",
                        help="negative control: permute each cell's raw frames before the window interpolation")
    parser.add_argument("--dedup", action="store_true",
                        help="after asymmetric cleaning, drop overlapping windows (see reach_event_clean.dedup_overlap)")
    parser.add_argument("--blocked", type=int, default=0, metavar="K",
                        help="outer split by time: per session, K contiguous blocks of events (2 = first/second half)")
    parser.add_argument("--label-shuffle", type=int, default=None, metavar="SEED",
                        help="negative control: segment-shuffle the labels per session (see segment_shuffle_labels)")
    parser.add_argument("--out", default=None, help="csv path for per-event outer-test predictions")
    args = parser.parse_args()

    _, events = build_events(stages=tuple(args.stages) if args.stages else None, dedup=args.dedup)
    if args.label_shuffle is not None:
        events = segment_shuffle_labels(events, seed=args.label_shuffle)
    tag = (f"stages={args.stages or 'all'} extras={args.extras or 'none'} norm={args.norm} "
           f"global={args.glob} general={args.general} model={args.model} skew={args.skew} dedup={args.dedup} "
           f"label_shuffle={args.label_shuffle} blocked={args.blocked}")
    print(f"[{tag}] sessions={events['token'].nunique()} events={len(events)} reach={events['label'].sum()}", flush=True)

    dec = Decoder(events, extras=tuple(args.extras), norm=args.norm, glob=args.glob, general=args.general,
                  model=args.model, skew=args.skew, blocked=args.blocked)
    print(f"  features: {dec.hand.shape[1]} handcrafted + {len(SEL_NAMES)} selectivity + {dec.extra.shape[1]} extra"
          f" + {len(GEN_NAMES) if args.general else 0} general", flush=True)
    folds = range(args.blocked) if args.blocked else (f - 1 for f in args.folds)
    res = dec.run(folds=tuple(folds))
    inner = np.mean([c["inner_macro_f1"] for c in dec.choices])
    print(f"  inner OOF macro F1 (mean of outer folds) = {inner:.3f}")
    summary = report(res)

    if args.out:
        res.to_csv(args.out, index=False)
        Path(args.out).with_suffix(".json").write_text(json.dumps(
            {"tag": tag, "choices": dec.choices, "inner_macro_f1": inner, **summary}, indent=2, default=float))
