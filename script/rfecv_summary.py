import os
from pathlib import Path
from dotenv import load_dotenv

from tools import SyncLogger
from src_predict import Summarizer

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
models      = ('LR', 'SVM', 'RF', 'XGB', 'LGB')
standard    = 3
rerun       = False
on_predict  = lambda x: True
src_folder  = PATH_DATA
sel_folder  = PATH_CACHE / Path(f"rfecv_init_vote/{beh}")
cac_folder  = PATH_CACHE / Path(f"rfecv_init_scor/{beh}")
tgt_folder  = PATH_PLOT  / Path("rfecv_predict/summary")

# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.rfecv_summary", message="start execute rfecv_summary.py")

    sm = Summarizer(
        seed=seed,
        src_folder=src_folder
    )

    sm.display_bar(
        sel_folder=sel_folder,
        cac_folder=cac_folder,
        tgt_folder=tgt_folder,
        beh=beh,
        on_predict=on_predict,
        models=models,
        standard=standard,
        rerun=rerun,
    )

    sm.display_pie(
        sel_folder=sel_folder,
        tgt_folder=tgt_folder,
        beh=beh,
        on_predict=on_predict,
        standard=standard,
    )

    logger.warning(role="script.rfecv_summary", message="complete execute rfecv_summary.py")