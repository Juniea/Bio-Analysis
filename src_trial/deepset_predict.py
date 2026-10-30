import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay

from tools import SyncLogger, check_folders, fetch_tokens
from .trial_predict import BEH_FPS, EXCLUDED, TASKS, _normalize, f1_scores


def build_spans(
    beh_path: str | os.PathLike | Path,
    sig_path: str | os.PathLike | Path,
    pre: float = 2.0,
    post: float = 2.0,
    max_len: int = 64,
    min_len: int = 0,
    extra_pre: float = 0.0,
    extra_post: float = 0.0,
    transform: str = "zscore",
    fixed_after: float | None = None,
    circular_shift: int | None = None,
    min_shift: int = 60,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Cut one window per behavior event, ranging [start - pre, end + post] seconds, sampled every second
    (or the fixed-length [start - pre, start + fixed_after] when fixed_after is given, ignoring the event end).
    Windows shorter than min_len points are extended by extra_pre / extra_post seconds,
    the rule only depends on the window length (never on the label) and applies to all events

    :param beh_path: The path to the behavior file
    :param sig_path: The path to the signal file
    :param pre: The seconds before the event start
    :param post: The seconds after the event end
    :param max_len: The most seconds kept for each window (longer windows are truncated at the tail)
    :param min_len: The windows with fewer points than this are extended
    :param extra_pre: The extra seconds before the start for the extended windows
    :param extra_post: The extra seconds after the end for the extended windows
    :param transform: The per-session signal transform (the signals are already deconvolved, sparse & non-negative):
                      "zscore", "log1p" (log1p then z-score), "binary" (> 0) or "smooth" (gaussian sigma 1 s then z-score)
    :param fixed_after: The seconds after the event start for a fixed-length window (post / extension are then unused)
    :param circular_shift: None for the real signals, or the seed of a circular-shift control: the whole signal of the
                           session (all cells together) is rolled in time by a random offset in [min_shift, n - min_shift]
                           samples, the behaviors stay in place; the offset only depends on the seed & the session
    :param min_shift: The smallest offset (samples, ~1 s each) of the circular shift

    :return: (event table [trial, behavior, start, end, extended, length], windows list of [n_cells, length]);
             the circular-shift offset is kept in events.attrs["shift"]
    """

    beh = pd.read_csv(beh_path)
    sig = pd.read_csv(sig_path)
    if "x0000" in sig.columns:
        sig = sig.set_index("x0000")

    # the same time conversions as src_predict.Preprocessor
    sig_fps = 1 / (1.000045 if Path(sig_path).stem.split('_')[0] == 'K215' else 1.001286)
    sig_time = sig.index.to_numpy(dtype=float) / sig_fps

    events = pd.DataFrame({
        "trial": beh.iloc[:, 0].ffill().fillna(0).astype(int),
        "behavior": beh["Behavior"].map(_normalize),
        "start": (beh["Start"] + BEH_FPS * 0.5) / BEH_FPS,
        "end": (beh["End"] + BEH_FPS * 0.5) / BEH_FPS,
    })
    events = events.loc[
        events["start"].notna() & events["end"].notna() & (events["end"] >= events["start"]) &
        ~events["behavior"].isin(EXCLUDED)
    ]
    if fixed_after is None:
        events["win_end"] = events["end"]
        n_points = np.floor(events["end"] - events["start"] + pre + post + 1e-6) + 1
        events["extended"] = n_points < min_len
        events["pre"] = pre + extra_pre * events["extended"]
        events["post"] = post + extra_post * events["extended"]
    else:
        events["win_end"] = events["start"] + fixed_after
        events["extended"] = False
        events["pre"] = pre
        events["post"] = 0.0
    events = events.loc[(events["start"] - events["pre"] >= sig_time[0]) & (events["win_end"] + events["post"] <= sig_time[-1])]
    events = events.reset_index(drop=True)

    # transform every cell over the whole session (unsupervised, no label used)
    values = sig.to_numpy(dtype=float)
    if transform == "binary":
        values = (values > 0).astype(float)
    else:
        if transform == "log1p":
            values = np.log1p(np.clip(values, 0, None))
        elif transform == "smooth":
            values = gaussian_filter1d(values, sigma=sig_fps * 1.0, axis=0, mode="nearest")
        elif transform != "zscore":
            raise ValueError(f"unknown transform: {transform}")
        values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-6)
    offset = 0
    if circular_shift is not None:
        rng = np.random.default_rng([circular_shift, zlib.crc32(Path(sig_path).stem.encode())])
        offset = int(rng.integers(min_shift, len(values) - min_shift))
        values = np.roll(values, offset, axis=0)

    windows = []
    for start, end, a, b in zip(events["start"], events["win_end"], events["pre"], events["post"]):
        stamps = np.arange(start - a, end + b + 1e-6, 1.0)[:max_len]
        windows.append(np.stack([np.interp(stamps, sig_time, values[:, c]) for c in range(values.shape[1])]).astype(np.float32))
    events["length"] = [w.shape[1] for w in windows]
    events = events.drop(columns=["pre", "post", "win_end"])
    events.attrs["shift"] = offset
    return events, windows


def _masked_conv(layers: nn.Sequential, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a stack of conv / activation / dropout layers, re-zeroing the padded seconds after every activation
    so that the padding never leaks into the real seconds through the following convolution

    :param h: [batch, channels, time]
    :param mask: [batch, 1, time], 1 for real seconds
    """
    for layer in layers:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            h = h * mask
    return h


def _encode_cells(phi: nn.Sequential, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    """
    The shared per-cell temporal encoder: [batch, cells, time] -> [batch, cells, channels, time]
    """
    b, n, t = x.shape
    mask = time_mask[:, None, :].repeat_interleave(n, dim=0)
    return _masked_conv(phi, x.reshape(b * n, 1, t), mask).reshape(b, n, -1, t)


class CellAttentionPool(nn.Module):
    """
    Multi-head gated attention pooling over the cells (Ilse et al. 2018), computed at every second,
    invariant to the cell number / order
    """

    def __init__(self, cell_dim: int = 32, att_dim: int = 32, heads: int = 4):
        super().__init__()
        self.V = nn.Linear(cell_dim, att_dim)
        self.U = nn.Linear(cell_dim, att_dim)
        self.w = nn.Linear(att_dim, heads)

    def forward(self, h: torch.Tensor, cell_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param h: [batch, cells, channels, time]
        :param cell_mask: [batch, cells], 1 for kept cells

        :return: (pooled [batch, heads * channels, time], weights [batch, cells, heads, time])
        """
        b, n, c, t = h.shape
        z = h.permute(0, 1, 3, 2)                                               # [b, n, t, c]
        score = self.w(torch.tanh(self.V(z)) * torch.sigmoid(self.U(z)))       # [b, n, t, heads]
        score = score.masked_fill(cell_mask[:, :, None, None] == 0, -1e4)
        weight = torch.softmax(score, dim=1).permute(0, 1, 3, 2)                # [b, n, heads, t]
        pooled = torch.einsum("bnkt,bnct->bkct", weight, h).reshape(b, -1, t)
        return pooled, weight


class DeepSetTCN(nn.Module):
    """
    Deep Sets over the cells (shared per-cell encoder + masked mean / attention pooling, invariant to the cell number / order),
    followed by a non-causal TCN over the time and masked mean & max pooling over the time
    """

    def __init__(
        self,
        cell_dim: int = 32,
        tcn_dim: int = 64,
        dilations: tuple = (1, 2, 4),
        dropout: float = 0.2,
        kernel_size: int = 3,
        pooling: str = "mean",
        heads: int = 4,
        n_out: int = 1,
    ):
        super().__init__()
        self.pooling = pooling
        self.n_out = n_out
        self.phi = nn.Sequential(
            nn.Conv1d(1, cell_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(cell_dim, cell_dim, 3, padding=1), nn.ReLU(),
        )

        # the pooled cell channels: mean -> cell_dim, attention -> heads * cell_dim, mean+attention -> both
        if pooling == "mean":
            pooled_dim = cell_dim
        elif pooling in ("attention", "mean+attention"):
            self.attention = CellAttentionPool(cell_dim=cell_dim, heads=heads)
            pooled_dim = heads * cell_dim + (cell_dim if pooling == "mean+attention" else 0)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        self.last_weight = None

        self.proj = nn.Conv1d(pooled_dim, tcn_dim, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(tcn_dim, tcn_dim, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(tcn_dim, tcn_dim, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.ReLU(), nn.Dropout(dropout),
            )
            for d in dilations
        ])
        self.head = nn.Sequential(nn.Linear(2 * tcn_dim, tcn_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(tcn_dim, n_out))

    def forward(self, x: torch.Tensor, cell_mask: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, cells, time]
        :param cell_mask: [batch, cells], 1 for real cells
        :param time_mask: [batch, time], 1 for real seconds

        :return: logits [batch] (n_out == 1) or [batch, n_out]
        """
        h = _encode_cells(self.phi, x, time_mask)

        # Deep Sets: masked mean and / or attention pooling over the cells
        pooled = []
        if self.pooling in ("mean", "mean+attention"):
            cm = cell_mask[:, :, None, None]
            pooled.append((h * cm).sum(dim=1) / cm.sum(dim=1).clamp(min=1))
        if self.pooling in ("attention", "mean+attention"):
            attended, self.last_weight = self.attention(h, cell_mask)
            pooled.append(attended)
        h = torch.cat(pooled, dim=1)

        # TCN over the time, padded seconds are kept at zero
        tm = time_mask[:, None, :]
        h = self.proj(h) * tm
        for block in self.blocks:
            h = (h + _masked_conv(block, h, tm)) * tm

        # masked mean & max pooling over the time
        mean = h.sum(dim=2) / tm.sum(dim=2).clamp(min=1)
        peak = h.masked_fill(tm == 0, -1e4).max(dim=2).values
        logits = self.head(torch.cat([mean, peak], dim=1))
        return logits.squeeze(1) if self.n_out == 1 else logits


class BertCellEncoder(nn.Module):
    """
    BERT-style encoder over the cells: every cell is one token (shared temporal conv encoder + masked mean & max
    pooling over the time), a learnable [CLS] token is prepended, padded cells are masked out of the self-attention
    and no positional encoding is used (the cells are unordered), so any cell number is accepted
    """

    def __init__(
        self,
        cell_dim: int = 32,
        d_model: int = 64,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Conv1d(1, cell_dim, 3, padding=1), nn.ReLU(),
            nn.Conv1d(cell_dim, cell_dim, 3, padding=1), nn.ReLU(),
        )
        self.embed = nn.Linear(2 * cell_dim, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, cell_mask: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, cells, time]
        :param cell_mask: [batch, cells], 1 for real cells
        :param time_mask: [batch, time], 1 for real seconds

        :return: logits [batch]
        """
        b = x.shape[0]
        h = _encode_cells(self.phi, x, time_mask)

        # one token per cell: masked mean & max pooling over the time
        tm = time_mask[:, None, None, :]
        mean = (h * tm).sum(dim=3) / tm.sum(dim=3).clamp(min=1)
        peak = h.masked_fill(tm == 0, -1e4).max(dim=3).values
        tokens = torch.cat([self.cls.expand(b, 1, -1), self.embed(torch.cat([mean, peak], dim=2))], dim=1)

        # the [CLS] token is never masked, the padded / dropped cells are ignored by the attention
        ignore = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), cell_mask == 0], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=ignore)
        return self.head(self.norm(out[:, 0])).squeeze(1)


def _collate(windows: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, ...]:
    n = max(w.shape[0] for w in windows)
    t = max(w.shape[1] for w in windows)
    x = np.zeros((len(windows), n, t), dtype=np.float32)
    cell_mask = np.zeros((len(windows), n), dtype=np.float32)
    time_mask = np.zeros((len(windows), t), dtype=np.float32)
    for i, w in enumerate(windows):
        x[i, :w.shape[0], :w.shape[1]] = w
        cell_mask[i, :w.shape[0]] = 1
        time_mask[i, :w.shape[1]] = 1
    return tuple(torch.from_numpy(a).to(device) for a in (x, cell_mask, time_mask))


def _split(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    One trial-grouped fold, stratified when both classes are frequent enough
    """
    if np.bincount(y, minlength=2).min() >= n_splits and len(np.unique(groups)) >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(cv.split(y, y, groups))
    return next(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(y, y, groups))


@torch.no_grad()
def _predict_proba(model: DeepSetTCN, windows: list[np.ndarray], device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(windows), batch_size):
        probs.append(torch.sigmoid(model(*_collate(windows[i:i + batch_size], device))).cpu().numpy())
    return np.concatenate(probs) if probs else np.empty(0)


def _best_threshold(y: np.ndarray, prob: np.ndarray, metric: str = "test_f1_macro") -> float:
    """
    The decision threshold maximizing the given F1 metric (key of f1_scores) on the given set
    """
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([f1_scores(y, (prob >= g).astype(int))[metric] for g in grid])])


def _metrics(real: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "test_accuracy": accuracy_score(real, pred),
        "test_balanced_accuracy": balanced_accuracy_score(real, pred) if len(np.unique(real)) > 1 else np.nan,
        **f1_scores(real, pred),
    }


@check_folders("tgt_folder")
def deepset_predict(
    src_folders: tuple[Path, ...],
    tgt_folder: str | os.PathLike | Path,
    task: str           = "cont",
    pre: float          = 2.0,
    post: float         = 2.0,
    max_len: int        = 64,
    min_len: int        = 0,
    extra_pre: float    = 0.0,
    extra_post: float   = 0.0,
    pooling: str        = "mean",
    arch: str           = "deepset_tcn",
    transform: str      = "zscore",
    split_by: str       = "trial",
    balance: str        = "weight",
    test_size: float    = 0.2,
    valid_size: float   = 0.2,
    cell_drop: float    = 0.2,
    epochs: int         = 100,
    patience: int       = 15,
    batch_size: int     = 64,
    lr: float           = 1e-3,
    weight_decay: float = 1e-3,
    min_class: int      = 10,
    seed: int           = 42,
) -> pd.DataFrame:
    """
    Event-level prediction with one Deep Sets + TCN model shared by all tokens:
    one sample per behavior event (window [start - pre, end + post]), trial-grouped train / valid / test split per token

    :param src_folders: The folders that contain the B & S csv files
    :param tgt_folder: The folder where the summary, the raw predictions, the history and the confusion matrix will be placed
    :param task: The key of TASKS that decides the positive behaviors
    :param pre: The seconds before the event start
    :param post: The seconds after the event end
    :param max_len: The most seconds kept for each window
    :param min_len: The windows with fewer points than this are extended (label-agnostic)
    :param extra_pre: The extra seconds before the start for the extended windows
    :param extra_post: The extra seconds after the end for the extended windows
    :param pooling: The cell pooling of DeepSetTCN: "mean", "attention" or "mean+attention"
    :param arch: The model: "deepset_tcn" (DeepSetTCN) or "bert" (BertCellEncoder)
    :param transform: The per-session signal transform: "zscore", "log1p", "binary" or "smooth" (see build_spans)
    :param balance: How the training classes are balanced (valid / test always keep the natural distribution):
                    "weight" (all events, positive class weighted in the loss),
                    "undersample_fixed" (one random negative subset as large as the positives, unweighted) or
                    "undersample_epoch" (a new random negative subset every epoch, unweighted)
    :param split_by: "trial" (all events of a trial on the same side) or "event" (events split independently,
                     stratified; neighbouring windows may overlap across train / test, so the scores are optimistic)
    :param test_size: The ratio of the testing set (per token, grouped by trial)
    :param valid_size: The ratio of the validation set inside the training set (early stopping & threshold)
    :param cell_drop: The ratio of cells randomly dropped for each training sample (augmentation)
    :param epochs: The most training epochs
    :param patience: The epochs without validation improvement before early stopping
    :param batch_size: The batch size
    :param lr: The learning rate
    :param weight_decay: The AdamW weight decay
    :param min_class: The least number of events of each class for a token to enter the per-token summary
    :param seed: The random seed

    :return: The per-token summary table
    """

    logger = SyncLogger(level="INFO")
    tag = f"{task}|{arch}|{pooling}|{transform}|split={split_by}|{balance}"    # shown on every log line to tell the runs apart
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the events of all tokens and split each token by trial
    tables, windows, sides = [], [], []
    for folder in src_folders:
        for token in fetch_tokens(folder=Path(folder), shard="all"):
            events, wins = build_spans(
                Path(folder) / f"{token}_B.csv", Path(folder) / f"{token}_S.csv",
                pre, post, max_len, min_len, extra_pre, extra_post, transform,
            )
            y = events["behavior"].isin(TASKS[task]).to_numpy(dtype=int)
            if split_by == "trial":
                groups = events["trial"].to_numpy()
            elif split_by == "event":
                groups = np.arange(len(events))     # every event is its own group -> plain stratified split
            else:
                raise ValueError(f"unknown split_by: {split_by}")

            side = np.full(len(y), "train", dtype=object)
            train_idx, test_idx = _split(y, groups, round(1 / test_size), seed)
            side[test_idx] = "test"
            sub_train, sub_valid = _split(y[train_idx], groups[train_idx], round(1 / valid_size), seed)
            side[train_idx[sub_valid]] = "valid"

            tables.append(events.assign(token=token, real=y, side=side))
            windows.extend(wins)
            sides.append(side)

    table = pd.concat(tables, ignore_index=True)
    y = table["real"].to_numpy()
    index = {s: np.flatnonzero(table["side"].to_numpy() == s) for s in ("train", "valid", "test")}
    logger.info(
        role="deepset_predict",
        message=f"[{tag}]{table['token'].nunique()} tokens | device={device} | "
                f"train={len(index['train'])} valid={len(index['valid'])} test={len(index['test'])} | "
                f"positive rate={y.mean():.3f}"
    )

    # model fitting with early stopping on the validation macro F1
    if arch == "deepset_tcn":
        model = DeepSetTCN(pooling=pooling).to(device)
    elif arch == "bert":
        model = BertCellEncoder().to(device)
    else:
        raise ValueError(f"unknown arch: {arch}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_pos = index["train"][y[index["train"]] == 1]
    train_neg = index["train"][y[index["train"]] == 0]
    if balance == "weight":
        pos_weight = torch.tensor(len(train_neg) / max(len(train_pos), 1), device=device)
    elif balance in ("undersample_fixed", "undersample_epoch"):
        pos_weight = torch.tensor(1.0, device=device)
        fixed_neg = rng.choice(train_neg, size=min(len(train_pos), len(train_neg)), replace=False)
    else:
        raise ValueError(f"unknown balance: {balance}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    valid_windows = [windows[i] for i in index["valid"]]
    monitor_windows = {s: [windows[i] for i in index[s]] for s in ("train", "test")}
    best_score, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        if balance == "weight":
            epoch_idx = index["train"]
        elif balance == "undersample_fixed":
            epoch_idx = np.concatenate([train_pos, fixed_neg])
        else:
            epoch_idx = np.concatenate([train_pos, rng.choice(train_neg, size=len(fixed_neg), replace=False)])
        order = rng.permutation(epoch_idx)
        losses, correct = [], 0
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            x, cell_mask, time_mask = _collate([windows[j] for j in batch], device)

            # augmentation: randomly drop cells, keeping at least one cell per sample
            keep = (torch.rand_like(cell_mask) >= cell_drop).float() * cell_mask
            keep[keep.sum(dim=1) == 0] = cell_mask[keep.sum(dim=1) == 0]

            target = torch.from_numpy(y[batch]).float().to(device)
            logits = model(x, keep, time_mask)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct += int(((logits.detach() > 0).float() == target).sum())

        # validation at the threshold maximizing the validation macro F1
        prob = _predict_proba(model, valid_windows, device)
        real = y[index["valid"]]
        threshold = _best_threshold(real, prob)
        pred = (prob >= threshold).astype(int)
        valid_f1 = f1_scores(real, pred)
        score = valid_f1["test_f1_macro"]

        # training & testing F1 at the validation threshold, for monitoring only (never used for stopping)
        monitor = {
            s: f1_scores(y[index[s]], (_predict_proba(model, monitor_windows[s], device) >= threshold).astype(int))
            for s in ("train", "test")
        }
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": correct / len(order),
            "valid_accuracy": accuracy_score(real, pred),
            "valid_balanced_accuracy": balanced_accuracy_score(real, pred),
            "valid_f1_macro": score,
            **{f"{s}_f1_{k}": f1[f"test_f1_{k}"] for s, f1 in (("train", monitor["train"]), ("valid", valid_f1),
                                                              ("test", monitor["test"])) for k in ("pos", "neg")},
        })
        logger.info(
            role="deepset_predict",
            message=f"[{tag}]epoch {epoch + 1:3d} | loss={history[-1]['train_loss']:.4f} | "
                    f"train acc={history[-1]['train_accuracy']:.3f} | "
                    f"valid acc={history[-1]['valid_accuracy']:.3f} f1_macro={score:.3f} | "
                    f"f1 {task}/other: "
                    f"train={history[-1]['train_f1_pos']:.3f}/{history[-1]['train_f1_neg']:.3f} "
                    f"valid={history[-1]['valid_f1_pos']:.3f}/{history[-1]['valid_f1_neg']:.3f} "
                    f"test={history[-1]['test_f1_pos']:.3f}/{history[-1]['test_f1_neg']:.3f}"
        )

        if score > best_score:
            best_score, wait = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    threshold = _best_threshold(y[index["valid"]], _predict_proba(model, valid_windows, device))
    logger.info(
        role="deepset_predict",
        message=f"[{tag}]stopped at epoch {len(history)} | best valid macro F1={best_score:.3f} | threshold={threshold:.2f}"
    )

    # control: predicting from the window length only (the annotated duration), same split & threshold rule
    length = np.log(table["length"].to_numpy(dtype=float))[:, None]
    control = LogisticRegression(class_weight="balanced").fit(length[index["train"]], y[index["train"]])
    control_threshold = _best_threshold(y[index["valid"]], control.predict_proba(length[index["valid"]])[:, 1])

    test = table.iloc[index["test"]].copy()
    test["probability"] = _predict_proba(model, [windows[i] for i in index["test"]], device)
    test["prediction"] = (test["probability"] >= threshold).astype(int)
    test["length_only"] = (control.predict_proba(length[index["test"]])[:, 1] >= control_threshold).astype(int)

    # per-token summary
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

    # pooled results over the testing events of all tokens
    pooled = pd.DataFrame([
        {"model": arch, **_metrics(test["real"].to_numpy(), test["prediction"].to_numpy())},
        {"model": "length_only", **_metrics(test["real"].to_numpy(), test["length_only"].to_numpy())},
        {"model": "majority", **_metrics(test["real"].to_numpy(), np.zeros(len(test), dtype=int))},
    ])

    summary.to_csv(Path(tgt_folder) / "summary.csv", index=False)
    pooled.to_csv(Path(tgt_folder) / "pooled.csv", index=False)
    test.to_csv(Path(tgt_folder) / "raw.csv", index=False)
    pd.DataFrame(history).to_csv(Path(tgt_folder) / "history.csv", index=False)
    torch.save(best_state, Path(tgt_folder) / "model.pt")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        test["real"],
        test["prediction"],
        display_labels=["Other", task],
        cmap="Reds",
        im_kw={"vmin": 0.0, "vmax": 1.0},
        normalize="true",
        ax=ax,
    )
    plt.savefig(Path(tgt_folder) / "matrix.pdf")
    plt.close()

    logger.info(role="deepset_predict", message=f"[{tag}]summary & pooled & raw & history & model saved to {tgt_folder}")
    return summary
