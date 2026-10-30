import os
import json
from pathlib import Path
from dotenv import load_dotenv

from tools import SyncLogger, fetch_tokens
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
beh             = "cont"
log_level       = "INFO"
sig_folder      = PATH_DATA
beh_folder      = PATH_DATA
tgt_folder      = PATH_CACHE / Path(f"sps_init/{beh}")
plt_bar_folder  = PATH_PLOT  / Path(f"sps_select/{beh}/bar")
plt_htm_folder  = PATH_PLOT  / Path(f"sps_select/{beh}/heatmap")
plt_cur_folder  = PATH_PLOT  / Path(f"sps_select/{beh}/curve")

# main execution logics
if __name__ == '__main__':
    logger = SyncLogger(level="INFO")
    logger.warning(role="script.sps_select_produce", message="start execute sps_select_produce.py")

    mapper = SPSSelector(
        sig_folder=sig_folder,
        beh_folder=beh_folder,
        seed=seed,
        log_level=log_level,
    )

    for token in fetch_tokens(folder=tgt_folder, shard="all"):

        with open(Path(tgt_folder) / Path(f"{token}_C.json"), 'r') as fp:
            lookup = json.load(fp)
            cells = lookup["selected_index"]

        mapper.event_heatmap(
            token=token,
            beh=beh,
            cells=cells,
            outer_width=8,
            inner_width=(-50, 50),
            cmap_sel="afmhot",
            cmap_oth="viridis",
            center_line=False,
            viz=False,
            plt_folder=plt_htm_folder,
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
            viz=False,
            plt_folder=plt_cur_folder,
        )

    logger.warning(role="script.sps_select_produce", message="complete execute sps_select_produce.py")