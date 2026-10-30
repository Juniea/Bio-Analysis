from tools import SyncLogger, fetch_tokens
from src_deduction import Locator, PenaltyComputer, Visualizer

import os
from pathlib import Path
from dotenv import load_dotenv

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory

# config the script parameters
shard           = "all"
seed            = 42
behs            = ("cont", "cont_inv", "both", "both_inv", "walk")
smooth          = 10
threshold       = 0.55
method          = 'max'
step            = 1
label_penalty   = 10.0
spike_penalty   = 0.1
rerun           = False
log_level       = "INFO"
sig_folder      = PATH_DATA
beh_folder      = PATH_DATA
plt_folder      = PATH_PLOT  / Path("deduce")
tgt_root        = PATH_CACHE / Path("deduce")

# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.deduce_3d", message="start execute deduce_3d.py")

    loc = Locator(
        sig_folder=sig_folder,
        beh_folder=beh_folder,
        log_level=log_level
    )

    window_configs = []
    for token in fetch_tokens(folder=sig_folder, shard=shard):
        for beh in behs:
            window_configs.append(loc.get_windows(
                token=token,
                beh=beh,
                smooth=smooth,
                threshold=threshold,
                method=method,
                step=step,
                rerun=rerun,
                tgt_folder=tgt_root / Path(beh)
            ))

    penalty_table = PenaltyComputer.compute_penalties(
        window_configs=tuple(window_configs),
        label_penalty = 10.0,
        spike_penalty = 0.1
    )
    Visualizer.display_3d_penalty(
        table=penalty_table,
        dx=0.15,
        dy=0.15,
        left_till=0,
        rotation=0,
        tgt_folder=plt_folder,
    )
    print()
    print(penalty_table)

    logger.warning(role="script.deduce_3d", message="start execute deduce_3d.py")