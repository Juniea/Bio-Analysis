from tools import SyncLogger
from src_deduction import Locator, Visualizer

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
token       = "K215_A_D1"
beh         = "cont"
smooth      = 10
threshold   = 0.55
method      = 'max'
step        = 1
rerun       = False
log_level   = "INFO"
sig_folder  = PATH_DATA
beh_folder  = PATH_DATA
tgt_folder  = PATH_CACHE / Path(f"deduce/{beh}")

# main execution logics
if __name__ == "__main__":
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.deduce_viz", message="start execute deduce_viz.py")

    loc = Locator(
        sig_folder=sig_folder,
        beh_folder=beh_folder,
        log_level=log_level
    )

    window_config = loc.get_windows(
        token=token,
        beh=beh,
        smooth=smooth,
        threshold=threshold,
        method=method,
        step=step,
        rerun=rerun,
        tgt_folder=tgt_folder
    )

    Visualizer.display_window(window_config['none'])
    Visualizer.display_window(window_config['both'])
    Visualizer.display_window(window_config['left'])
    Visualizer.display_window(window_config['right'])

    logger.warning(role="script.deduce_viz", message="complete execute deduce_viz.py")