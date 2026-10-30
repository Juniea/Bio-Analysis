import os
from pathlib import Path
from dotenv import load_dotenv

from tools import SyncLogger
from src_select import SPSSelector

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory

# config the script parameters
seed            = 42
token           = "K215_A_D1"
beh             = "cont"
standard        = 4
threshold       = 0.60
score           = "f1"
resize          = True
flip            = False
rerun           = False
viz             = True
log_level       = "INFO"
sig_folder      = PATH_DATA
beh_folder      = PATH_DATA
tgt_folder      = PATH_CACHE / Path(f"sps_init/{beh}")
plt_bar_folder  = PATH_PLOT / Path(f"sps_select/{beh}/bar")
plt_res_folder  = PATH_PLOT / Path(f"sps_select/{beh}/heatmap")

# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.sps_select_adjust", message="start execute sps_select_adjust.py")

    mapper = SPSSelector(
        sig_folder=sig_folder,
        beh_folder=beh_folder,
        seed=seed,
        log_level=log_level,
    )

    cells = mapper.sps_select(
        token=token,
        beh=beh,
        standard=standard,
        threshold=threshold,
        score=score,
        resize=resize,
        flip=flip,
        rerun=rerun,
        viz=viz,
        tgt_folder=tgt_folder,
        plt_folder=plt_bar_folder,
    )

    mapper.event_heatmap(
        token=token,
        beh=beh,
        cells=cells,
        outer_width=8,
        inner_width=(-50, 50),
        cmap_sel="afmhot",
        cmap_oth="viridis",
        center_line=False,
        viz=viz,
        plt_folder=plt_res_folder,
    )

    mapper.event_curve(
        token=token,
        beh=beh,
        cells=cells,
        outer_width=8,
        inner_width=(-50, 50),
        cmap_sel="crimson",
        cmap_oth="royalblue",
        center_line=False,
        viz=viz,
        plt_folder=plt_res_folder,
    )

    logger.warning(role="script.sps_select_adjust", message="complete execute sps_select_adjust.py")