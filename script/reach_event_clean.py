import re
from pathlib import Path

import numpy as np
import pandas as pd

# config the script parameters
root        = Path(__file__).resolve().parents[1]
src_folders = (root / Path("data/initialization"), root / Path("data/memory"))
stages      = None   # e.g. ("D1", "D2", "D7", "D8", "M4", "M5"); None keeps every stage
win_pre     = 2.0
win_post    = 4.0
BEH_FPS     = 7.06

MARKERS = {"valve opened", "two photon imaging", "two-photon imaging", "last 2p", "last 2p frame", "last frame"}
REACH = {"cont reach", "cont reaching"}


def normalize(name: str) -> str:
    """
    lower-case, collapse whitespace, drop the parenthesis and everything after it
    """
    name = str(name).split("(")[0]
    return re.sub(r"\s+", " ", name).strip().lower()


def load_events(beh_path: Path, sig_path: Path) -> pd.DataFrame:
    token = beh_path.stem[:-2]
    mouse, fov, stage = token.split("_")

    # the first column is the trial id, but its header varies ("Trial", "Trial #", "Trial ", "n")
    beh = pd.read_csv(beh_path)
    beh = beh.rename(columns={beh.columns[0]: "Trial"})
    beh["Trial"] = beh["Trial"].ffill()
    beh["behavior"] = beh["Behavior"].map(normalize)
    beh["start"] = (beh["Start"] + BEH_FPS * 0.5) / BEH_FPS
    beh["end"] = (beh["End"] + BEH_FPS * 0.5) / BEH_FPS

    sig_fps = 1 / 1.000045 if mouse == "K215" else 1 / 1.001286
    frames = pd.read_csv(sig_path, usecols=["x0000"])["x0000"].to_numpy(dtype=float)
    sig_lo, sig_hi = frames.min() / sig_fps, frames.max() / sig_fps

    ev = beh.loc[~beh["behavior"].isin(MARKERS)]
    ev = ev.dropna(subset=["start", "end"])
    ev = ev.loc[ev["end"] >= ev["start"]]
    ev = ev.loc[(ev["start"] - win_pre >= sig_lo) & (ev["start"] + win_post <= sig_hi)]

    ev = ev.assign(token=token, mouse=mouse, fov=fov, stage=stage, label=ev["behavior"].isin(REACH).astype(int))
    return ev.loc[:, ["token", "mouse", "fov", "stage", "Trial", "behavior", "start", "end", "label"]].reset_index(drop=True)


def asymmetric_clean(ev: pd.DataFrame) -> pd.DataFrame:
    """
    window_any: drop a non-reach event if [start - 2, start + 4] overlaps (length > 0) any reach in the same session
    """
    reach = ev.loc[ev["label"] == 1, ["start", "end"]].to_numpy()
    w_lo = ev["start"].to_numpy()[:, None] - win_pre
    w_hi = ev["start"].to_numpy()[:, None] + win_post
    overlap = np.minimum(w_hi, reach[:, 1]) - np.maximum(w_lo, reach[:, 0])
    hit = (overlap > 0).any(axis=1) if len(reach) else np.zeros(len(ev), dtype=bool)
    return ev.loc[~(hit & (ev["label"] == 0).to_numpy())]


def dedup_overlap(ev: pd.DataFrame) -> pd.DataFrame:
    """
    drop window overlaps ([start - 2, start + 4] of two events share time, i.e. starts < 6 s apart), per session:
    reach is always kept; a non-reach overlapping any reach is dropped; when two non-reach overlap, the later
    one is dropped (dropped events take no further part in the comparison)
    """
    gap = win_pre + win_post
    s, y = ev["start"].to_numpy(), ev["label"].to_numpy()
    keep = y == 1
    for i in np.argsort(s, kind="stable"):
        if y[i] == 0 and not (keep & (np.abs(s - s[i]) < gap)).any():
            keep[i] = True
    return ev.loc[keep]


def build_events(stages: tuple[str, ...] | None = None, dedup: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, clean = [], []
    for folder in src_folders:
        for beh_path in sorted(folder.glob("*_B.csv")):
            if stages is not None and beh_path.stem.split("_")[2] not in stages:
                continue
            ev = load_events(beh_path, beh_path.with_name(beh_path.stem[:-2] + "_S.csv"))
            raw.append(ev)
            ev = asymmetric_clean(ev)
            clean.append(dedup_overlap(ev) if dedup else ev)
    return pd.concat(raw, ignore_index=True), pd.concat(clean, ignore_index=True)


def report(tag: str, raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    n_drop = len(raw) - len(clean)
    print(f"[{tag}] sessions={raw['token'].nunique()}")
    print(f"  before: {len(raw)} events, reach {raw['label'].sum()}, non-reach {(raw['label'] == 0).sum()}")
    print(f"  removed non-reach: {n_drop}")
    print(f"  after : {len(clean)} events, reach {clean['label'].sum()} ({clean['label'].mean():.1%}), "
          f"non-reach {(clean['label'] == 0).sum()}")


if __name__ == '__main__':
    raw, clean = build_events(stages=None)
    report("all stages", raw, clean)

    sub = ("D1", "D2", "D7", "D8", "M4", "M5")
    raw_s, clean_s = build_events(stages=sub)
    report("D1 D2 D7 D8 M4 M5", raw_s, clean_s)
    print(clean_s.groupby(["stage", "fov"])["label"].agg(events="size", reach="sum").to_string())
