from pathlib import Path

import pandas as pd

from tools import SyncLogger
from src_trial import lgbm_nested

# path of this project directory (config/.env.path points to another machine)
PATH_ROOT = Path(__file__).resolve().parents[1]
PATH_DATA = PATH_ROOT / "data/initialization"
PATH_MEMO = PATH_ROOT / "data/memory"
PATH_PLOT = PATH_ROOT / "plot"

# config the script parameters: the no_min+ext_selectivity pipeline (fixed 2+4 window, 25 selectivity features),
# trained & tested on some stages only, with the non-reaches cleaned by window_any and then kept only once when their
# windows overlap each other (the first one stays, see first_negatives);
# train_only_stages: the other stages added to the training part of every fold, never tested (() to leave them out)
seed            = 42
pre, fixed_after = 2.0, 4.0
transform       = "zscore"
feature_set     = "extended"
regularization  = "regularized"
drop_overlap    = "window_any_dedup"
split_bys       = ("event",)    # one run each: "event" (free), "trial", or "block": 10 contiguous time blocks per session
circular_shifts = (None,)       # one run each: None is the real data, an int the seed of the circular-shift control (shifts.csv)
feature_windows = (None,)       # one run each: None sees -2..+4, (-2, 0) / (-3, -1) only those seconds (same events)
clean_window    = None          # None: clean with -2..+4; (-3, -1): every overlap rule judged on -3..-1 instead
run_tag         = "_no_noise"   # appended to the output folder: "noise generation" is excluded from the events (EXCLUDED)
stages          = ("D1", "D2", "D7", "D8", "M4", "M5")
train_only_stages = ()
select_by       = "f1_macro"    # the inner out-of-fold score choosing the trees & threshold: "f1_macro" or "f1_pos" (reach F1)
outer_folds     = 5
inner_folds     = 5
max_trees       = 1000
src_folders     = (PATH_DATA, PATH_MEMO)
task            = "cont"
weighting       = "global"
variants        = {             # name: (drop_prefixes, selectivity, cell_selection)
    "no_min+ext_selectivity": (("min_",), "extended", "none"),
    "no_min+ext_selectivity+sps_fixed": (("min_",), "extended", "sps_fixed"),
}
run_variants    = ("no_min+ext_selectivity",)

# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.lgbm_stages", message="start execute lgbm_stages.py")

    jobs = [(n, s, c, f) for n in run_variants for s in split_bys for c in circular_shifts for f in feature_windows]
    for name, split_by, circular_shift, feature_window in jobs:
        base_folder = PATH_PLOT / Path(f"lgbm_nested/{task}/fixed{pre:g}+{fixed_after:g}_{drop_overlap}_split_{split_by}"
                                       f"_{'+'.join(stages)}{run_tag}")
        drop_prefixes, selectivity, cell_selection = variants[name]
        folder = base_folder / (f"{name}+train_{'+'.join(train_only_stages)}" if train_only_stages else name)
        if circular_shift is not None:
            folder = base_folder / "circular_shift" / f"{folder.name}+shift{circular_shift}"
        if select_by != "f1_macro":
            folder = folder.with_name(f"{folder.name}+select_{select_by}")
        if feature_window is not None:
            folder = folder.with_name(f"{folder.name}+features_{feature_window[0]:+g}..{feature_window[1]:+g}")
        if clean_window is not None:
            folder = folder.with_name(f"{folder.name}+clean_{clean_window[0]:+g}..{clean_window[1]:+g}")
        if (folder / "per_token.csv").exists():
            logger.info(role="script.lgbm_stages", message=f"[{name}] already done, reuse {folder}")
        else:
            lgbm_nested(
                src_folders=src_folders,
                tgt_folder=folder,
                task=task,
                pre=pre,
                transform=transform,
                split_by=split_by,
                feature_set=feature_set,
                fixed_after=fixed_after,
                regularization=regularization,
                drop_overlap=drop_overlap,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                max_trees=max_trees,
                drop_prefixes=drop_prefixes,
                selectivity=selectivity,
                weighting=weighting,
                cell_selection=cell_selection,
                stages=stages,
                train_only_stages=train_only_stages,
                select_by=select_by,
                feature_window=feature_window,
                clean_window=clean_window,
                circular_shift=circular_shift,
                seed=seed,
            )
        pooled = pd.read_csv(folder / "pooled.csv").set_index("scope")
        logger.info(role="script.lgbm_stages", message=f"[split={split_by} shift={circular_shift}] {folder.name} pooled\n"
                                                       f"{pooled.round(3).to_string()}")
    logger.warning(role="script.lgbm_stages", message="complete execute lgbm_stages.py")
