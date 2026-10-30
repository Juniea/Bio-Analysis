import os
from pathlib import Path
from dotenv import load_dotenv

from tools import SyncLogger
from src_predict import rfecv_select

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory

# config the script parameters
seed        = 42
beh         = "cont"
on_select   = lambda x: True
tt_ratio    = (0.5, 0.5)
rerun       = False
src_folder  = PATH_DATA
tgt_folder  = PATH_CACHE / Path(f'rfecv_init_vote/{beh}')

# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.rfecv_select", message="start execute rfecv_select.py")

    rfecv_select(
        src_folder=src_folder,
        tgt_folder=tgt_folder,
        beh=beh,
        on_select=on_select,
        tt_ratio=tt_ratio,
        rerun=rerun,
        seed=seed,
    )

    logger.warning(role="script.rfecv_select", message="complete execute rfecv_select.py")