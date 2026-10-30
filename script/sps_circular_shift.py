import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_event_clean import build_events, src_folders
from reach_event_decode import make_folds, OFFSETS

warnings.filterwarnings("ignore")

# config the script parameters
n_shift     = 500       # random circular shifts per cell; 0 = every possible shift (exhaustive)
min_shift   = 30.0      # seconds; shifts closer than this to 0 (mod length) are excluded
alpha       = 0.05
seed        = 42


def interp_index(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    linear-interpolation indices / weights of query times q on the frame times t (same result as np.interp)
    """
    j = np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(t) - 2)
    a = np.clip((q - t[j]) / (t[j + 1] - t[j]), 0.0, 1.0)
    return j, a


def lda_cv_f1(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """
    single-cell shrinkage LDA (equal priors), cross-validated reach F1, batched over the leading axis

    :param x: (shifts, events, 7) windows of ONE cell under every shift
    :param y: (events,) labels
    :param fold: (events,) fold ids
    :return: (shifts,) reach F1
    """
    s, _, d = x.shape
    tp = np.zeros(s)
    fp = np.zeros(s)
    fn = np.zeros(s)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        y_tr, y_te = y[tr], y[te]
        if (y_tr == 1).sum() < 2 or (y_tr == 0).sum() < 2:
            pred = np.zeros((s, te.sum()), dtype=bool)
        else:
            x_tr = x[:, tr]
            mu1 = x_tr[:, y_tr == 1].mean(1)
            mu0 = x_tr[:, y_tr == 0].mean(1)
            c = np.concatenate([x_tr[:, y_tr == 1] - mu1[:, None], x_tr[:, y_tr == 0] - mu0[:, None]], axis=1)
            cov = np.einsum("snd,sne->sde", c, c) / (len(y_tr) - 2)
            ridge = 0.1 * np.trace(cov, axis1=1, axis2=2) / d + 1e-6
            cov = cov + ridge[:, None, None] * np.eye(d)[None]
            w = np.linalg.solve(cov, (mu1 - mu0)[..., None])[..., 0]
            b = -(w * (mu1 + mu0) / 2).sum(-1)
            pred = np.einsum("snd,sd->sn", x[:, te], w) + b[:, None] > 0
        pos = (y_te == 1)[None]
        tp += (pred & pos).sum(1)
        fp += (pred & ~pos).sum(1)
        fn += (~pred & pos).sum(1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def bh_reject(p: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    passed = np.zeros(len(p), dtype=bool)
    hits = np.flatnonzero(ranked <= q)
    if len(hits):
        passed[order[: hits.max() + 1]] = True
    return passed


def session_cells(token: str, ev: pd.DataFrame) -> pd.DataFrame:
    path = next(p for f in src_folders for p in f.glob(f"{token}_S.csv"))
    sig = pd.read_csv(path)
    fps = 1 / 1.000045 if token.startswith("K215") else 1 / 1.001286
    t = sig.pop("x0000").to_numpy(float) / fps
    names = sig.columns.to_list()
    x = sig.to_numpy(float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    T = len(t)

    y = ev["label"].to_numpy()
    fold = make_folds(y, seed)
    j, a = interp_index(t, ev["start"].to_numpy()[:, None] + OFFSETS[None, :])

    rng = np.random.default_rng(seed)
    lo = int(np.ceil(min_shift / np.median(np.diff(t))))
    null_shifts = np.arange(lo, T - lo + 1) if n_shift == 0 else rng.integers(lo, T - lo, n_shift)
    shifts = np.r_[0, null_shifts]                               # shift 0 = the real alignment
    idx = (j[None] - shifts[:, None, None]) % T                  # rolled signal: x'[m] = x[(m - k) % T]
    idx1 = (j[None] + 1 - shifts[:, None, None]) % T

    rows = []
    for c, name in enumerate(names):
        xc = x[:, c]
        win = (1 - a)[None] * xc[idx] + a[None] * xc[idx1]       # (shifts + 1, events, 7)
        f1 = lda_cv_f1(win, y, fold)
        real, null = f1[0], f1[1:]
        rows.append(dict(
            token=token, cell=name, real_f1=real, null_mean=null.mean(), null_95=np.quantile(null, 0.95),
            p=(1 + (null >= real).sum()) / (1 + len(null)),
        ))
    out = pd.DataFrame(rows)
    out["selected"] = out["p"] <= alpha
    out["selected_bh"] = bh_reject(out["p"].to_numpy(), alpha)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=["D1", "D2", "D7", "D8", "M4", "M5"])
    parser.add_argument("--out", default=None, help="csv path for the per-cell results")
    parser.add_argument("--n-shift", type=int, default=n_shift, help="random shifts per cell; 0 = exhaustive")
    args = parser.parse_args()
    n_shift = args.n_shift

    _, events = build_events(stages=tuple(args.stages))
    results = []
    for token, ev in events.groupby("token", sort=True):
        if ev["label"].sum() == 0:
            print(f"  {token}: no reach events, skipped", flush=True)
            continue
        r = session_cells(token, ev.reset_index(drop=True))
        results.append(r)
        print(f"  {token}: cells={len(r)} selected={r['selected'].sum()} (BH {r['selected_bh'].sum()}) "
              f"reach={int(ev['label'].sum())}", flush=True)
    res = pd.concat(results, ignore_index=True)
    if args.out:
        res.to_csv(args.out, index=False)

    res[["mouse", "fov", "stage"]] = res["token"].str.split("_", expand=True)
    summary = res.groupby(["mouse", "fov", "stage"]).agg(
        cells=("cell", "size"), selected=("selected", "sum"), selected_bh=("selected_bh", "sum"),
    )
    summary["frac"] = summary["selected"] / summary["cells"]
    summary["expected_by_chance"] = summary["cells"] * alpha
    pd.set_option("display.width", 200)
    print("\nper mouse / FOV / stage:")
    print(summary.round(2).to_string())
    print(f"\ntotal: {res['selected'].sum()} / {len(res)} cells selected at p<={alpha} "
          f"(chance ~{alpha * len(res):.0f}); BH-FDR: {res['selected_bh'].sum()}")
