from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import fisher_exact

from tools import SyncLogger, fetch_tokens
from src_trial.deepset_predict import build_spans
from src_trial.trial_predict import TASKS
from src_trial.lgbm_cellstats import overlap_mask, first_negatives
from src_trial.cell_select import sps_select

# path of this project directory (config/.env.path points to another machine)
PATH_ROOT = Path(__file__).resolve().parents[1]
PATH_DATA = PATH_ROOT / "data/initialization"
PATH_MEMO = PATH_ROOT / "data/memory"
PATH_PLOT = PATH_ROOT / "plot"

# config the script parameters: the cells that tell a reach from the other behaviors BEFORE its start (PRE, the points
# -3..-1 s) vs AFTER it (POST, the points 0..+4 s), each by the SPS rule on that window alone (5 voters, macro F1 by a
# 5-fold event cross-validation, >= standard votes at >= threshold, all events of the session at once, no fallback);
# the events are those of the six-stage framework (fixed -2..+4 window, window_any_dedup cleaning)
task            = "cont"
stages          = ("D1", "D2", "D7", "D8", "M4", "M5")
pre, fixed_after = 2.0, 4.0     # the cleaning window, as the decoder
pre_window      = (-3, -1)      # seconds around the start seen by the PRE selection
post_window     = (0, 4)        # seconds seen by the POST selection
view            = (5.0, 5.0)    # seconds before / after the start drawn in the heatmaps
threshold       = 0.58
standard        = 4
outer_width     = 6             # cells per row in the session figures
tgt_folder      = PATH_PLOT / "pre_reach_cells"
src_folders     = (PATH_DATA, PATH_MEMO)


def cut(wins: np.ndarray, first: float, last: float, start_pre: float) -> np.ndarray:
    """The points first..last seconds of windows that start start_pre seconds before the event start"""
    return wins[:, :, int(round(start_pre + first)):int(round(start_pre + last)) + 1]


def select(scores: np.ndarray) -> np.ndarray:
    return (scores >= threshold).sum(axis=1) >= standard


def process(folder: Path, token: str) -> tuple[pd.DataFrame, dict] | None:
    sig, beh = folder / f"{token}_S.csv", folder / f"{token}_B.csv"
    build_pre = max(pre, -pre_window[0])
    events, wins = build_spans(beh, sig, build_pre, 2.0, 64, fixed_after=fixed_after)
    y = events["behavior"].isin(TASKS[task]).to_numpy(dtype=int)
    keep = ~(y == 0) | ~overlap_mask(events, y, "window_any", pre, 2.0, fixed_after)
    keep &= first_negatives(events, y, keep, pre, 2.0, fixed_after)
    events, w, y = events.loc[keep].reset_index(drop=True), np.stack(wins)[keep], y[keep]
    if min(y.sum(), (1 - y).sum()) < 2:
        return None
    names = pd.read_csv(sig, nrows=0).columns.drop("x0000")
    _, pre_info = sps_select(cut(w, *pre_window, build_pre), y, pre=2.0, threshold=threshold, standard=standard, adaptive=False)
    _, post_info = sps_select(cut(w, *post_window, build_pre), y, pre=0.0, threshold=threshold, standard=standard, adaptive=False)
    cells = pd.DataFrame({"token": token, "cell": names, "pre_score": pre_info["scores"].mean(axis=1),
                          "post_score": post_info["scores"].mean(axis=1), "pre_selected": select(pre_info["scores"]),
                          "post_selected": select(post_info["scores"]), "n_reach": int(y.sum()), "n_other": int((1 - y).sum())})

    # the display windows (view) of the same events
    d_events, d_wins = build_spans(beh, sig, view[0], 2.0, 64, fixed_after=view[1])
    d_events = d_events.reset_index(drop=True).assign(idx=lambda x: np.arange(len(x)))
    m = events[["start"]].merge(d_events[["start", "idx"]], on="start", how="inner")
    disp = np.stack(d_wins)[m["idx"].to_numpy()]
    dy = events.set_index("start").loc[m["start"], "behavior"].isin(TASKS[task]).to_numpy()
    return cells, {"disp": disp, "y": dy}


def draw(token: str, cells: pd.DataFrame, data: dict, group: str) -> None:
    """One figure of the PRE or POST cells of a token, in tgt_folder/<group>"""
    own, other_group = ("pre", "post") if group == "PRE" else ("post", "pre")
    window = pre_window if group == "PRE" else post_window
    picked = cells.loc[cells[f"{own}_selected"]].sort_values(f"{own}_score", ascending=False)
    t = np.arange(-view[0], view[1] + 1)
    n = max(len(picked), 1)
    cols = min(outer_width, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(2 * rows, cols, figsize=(3.1 * cols, 4.6 * rows), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1] * rows})
    for ax in axes.ravel():
        ax.axis("off")
    for k, (_, c) in enumerate(picked.iterrows()):
        i = int(np.flatnonzero(cells["cell"].to_numpy() == c["cell"])[0])
        r, col = 2 * (k // cols), k % cols
        reach, other = data["disp"][data["y"], i, :], data["disp"][~data["y"], i, :]
        ax = axes[r, col]
        ax.axis("on")
        ax.imshow(reach, aspect="auto", cmap="afmhot", extent=(t[0] - 0.5, t[-1] + 0.5, len(reach), 0),
                  vmin=np.percentile(reach, 1), vmax=np.percentile(reach, 98))
        ax.axvline(0, color="white", linestyle="--", linewidth=1)
        ax.axvspan(window[0] - 0.5, window[1] + 0.5, color="cyan", alpha=0.12)
        both = "PRE + POST" if c[f"{other_group}_selected"] else f"{group} only"
        ax.set_title(f"{c['cell']}  {both}\nPRE F1 {c['pre_score']:.2f} | POST F1 {c['post_score']:.2f}", fontsize=8)
        ax.set_ylabel("reach events", fontsize=7)
        ax.tick_params(labelsize=6)
        ax = axes[r + 1, col]
        ax.axis("on")
        for v, color, label in ((reach, "crimson", f"reach ({len(reach)})"), (other, "royalblue", f"non-reach ({len(other)})")):
            mu, se = v.mean(axis=0), v.std(axis=0) / np.sqrt(max(len(v), 1))
            ax.plot(t, mu, color=color, label=label, linewidth=1.2)
            ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.2)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvspan(window[0] - 0.5, window[1] + 0.5, color="cyan", alpha=0.12)
        ax.set_xlabel("seconds from event start", fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, frameon=False)
    n_pre, n_post = int(cells["pre_selected"].sum()), int(cells["post_selected"].sum())
    n_both = int((cells["pre_selected"] & cells["post_selected"]).sum())
    fig.suptitle(f"{token} - {group} cells ({window[0]:+d}..{window[1]:+d} s, shaded): {len(picked)} of {len(cells)} | "
                 f"PRE {n_pre}, POST {n_post}, both {n_both} | reach {cells['n_reach'].iloc[0]}, "
                 f"non-reach {cells['n_other'].iloc[0]} | rule: macro F1 >= {threshold} in >= {standard}/5 voters",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    (tgt_folder / group).mkdir(exist_ok=True)
    fig.savefig(tgt_folder / group / f"{token}_{group}_cells.pdf")
    plt.close(fig)


# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.pre_reach_cells", message="start execute pre_reach_cells.py")
    tgt_folder.mkdir(parents=True, exist_ok=True)
    jobs = [(folder, token) for folder in src_folders for token in fetch_tokens(folder=folder, shard="all")
            if token.split("_")[-1] in stages]
    results = Parallel(n_jobs=16)(delayed(process)(f, t) for f, t in jobs)

    rows = []
    for (_, token), res in zip(jobs, results):
        if res is None:
            logger.info(role="script.pre_reach_cells", message=f"{token}: a class is missing, skipped")
            continue
        cells, data = res
        draw(token, cells, data, "PRE")
        draw(token, cells, data, "POST")
        rows.append(cells)
        a, b = cells["pre_selected"].to_numpy(), cells["post_selected"].to_numpy()
        logger.info(role="script.pre_reach_cells",
                    message=f"{token}: PRE {a.sum()} | POST {b.sum()} | both {(a & b).sum()} of {len(cells)}")
    cells = pd.concat(rows, ignore_index=True)
    cells.to_csv(tgt_folder / "pre_post_cells.csv", index=False)

    # per FOV x stage counts & the overlap test (Fisher exact, PRE vs POST within the session)
    summary = []
    for token, g in cells.groupby("token", sort=False):
        a, b = g["pre_selected"].to_numpy(), g["post_selected"].to_numpy()
        table = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
        mouse, fov, stage = token.split("_")
        summary.append({"token": token, "mouse": mouse, "fov": fov, "stage": stage, "cells": len(g),
                        "reach": g["n_reach"].iloc[0], "non_reach": g["n_other"].iloc[0], "pre": a.sum(), "post": b.sum(),
                        "both": (a & b).sum(), "pre_only": (a & ~b).sum(), "post_only": (~a & b).sum(),
                        "fisher_p": fisher_exact(table, alternative="greater")[1]})
    summary = pd.DataFrame(summary)
    summary.to_csv(tgt_folder / "pre_post_summary.csv", index=False)

    # the overview figure: PRE only / both / POST only per FOV x stage
    order = {"D1": 0, "D2": 0, "D7": 1, "D8": 1, "M4": 2, "M5": 2}
    s = summary.assign(o=summary["stage"].map(order)).sort_values(["mouse", "fov", "o"])
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(s)), 5))
    x = np.arange(len(s))
    ax.bar(x, s["pre_only"], color="teal", label="PRE only (-3..-1 s)")
    ax.bar(x, s["both"], bottom=s["pre_only"], color="purple", label="PRE + POST")
    ax.bar(x, s["post_only"], bottom=s["pre_only"] + s["both"], color="orange", label="POST only (0..+4 s)")
    ax.plot(x, s["cells"], "k_", markersize=12, label="cells in the FOV")
    ax.set_xticks(x, s["token"], rotation=90, fontsize=8)
    ax.set_ylabel("cells")
    ax.set_title(f"cells telling a reach apart before (PRE) / after (POST) its start, macro F1 >= {threshold}, >= {standard}/5")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(tgt_folder / "pre_post_overview.pdf")
    plt.close(fig)
    logger.info(role="script.pre_reach_cells", message=f"\n{summary.drop(columns=['mouse', 'fov', 'stage']).round(3).to_string(index=False)}")
    logger.warning(role="script.pre_reach_cells", message="complete execute pre_reach_cells.py")
