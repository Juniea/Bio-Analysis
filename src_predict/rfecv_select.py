from typing import Callable, Tuple

import os
from pathlib import Path
from dotenv import load_dotenv

from tools import ClassificationFeatureEngineer, SyncLogger, check_folders, fetch_tokens
from .preprocessor import Preprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


@check_folders("src_folder", "tgt_folder")
def rfecv_select(
    src_folder: str | os.PathLike | Path,
    tgt_folder: str | os.PathLike | Path,
    beh: str                        = "cont",
    on_select: Callable             = lambda x: True,
    tt_ratio: Tuple[float, float]   = (0.5, 0.5),
    rerun: bool                     = False,
    seed: int                       = 42
) -> None:
    """
    Generate the voting & config of RFECV-SELECT for all desired tokens

    :param src_folder:  The source folder for data
    :param tgt_folder:  The cache folder for the voting-config pair
    :param beh:         The interested behavior
    :param on_select:   The extra function for filtering and get the desired tokens only
    :param tt_ratio:    The ratio between the train dataset size and test dataset size
    :param rerun:       The toggle for force-rerunning the RFECV-SELECT even if the cached results already exist
    :param seed:        The global random seed

    :return: None
    """

    logger = SyncLogger(level='INFO')

    # generate all paths for data files to-be-processed
    tokens = fetch_tokens(folder=src_folder, shard="all")
    tokens = tuple(filter(on_select, tokens))

    pool = tuple([
        (token, Path(src_folder) / Path(f"{token}_B.csv"), Path(src_folder) / Path(f"{token}_S.csv"))
        for token in tokens
    ])

    # iterate through all tokens for RFECV-SELECT and store the results in cache
    for tok, beh_file, sig_file in pool:

        if (
                (not rerun) and
                (Path(tgt_folder) / Path(f"{tok}_V.csv")).exists() and
                (Path(tgt_folder) / Path(f"{tok}_P.json")).exists()
        ):
            logger.info(role="rfecv_select", message=f"{tok}'s voting & config already exists.")
            continue

        try:
            prep = Preprocessor(
                beh_path=beh_file,
                sig_path=sig_file,
                target=beh,
                tvt_ratio=(tt_ratio[0], 0.0, tt_ratio[1]),
                shuffle=True,
                normalizer='z-score',
                agg=False,
                smote='auto',
                resize=True,
                log_level="WARNING"
            )
            x_train = prep.get_train(shard='x')
            y_train = prep.get_train(shard='y')

            eng = ClassificationFeatureEngineer(x_train, y_train, seed)
            selected, votes = eng.RFECV_select(
                models=['LR', 'LGB', 'XGB', 'RF', 'SVM'],
                cv=10,
                normalize='z-score',
                standard=4,
                verbose=False,
            )

        except Exception as e:
            logger.warning(role="rfecv_select", message=f"{tok} failed ({str(e).strip()})")
            continue

        # store the selection result in cache
        try:
            votes.to_csv(Path(tgt_folder) / Path(f"{tok}_V.csv"), index=False)
            prep.json(Path(tgt_folder) / Path(f"{tok}_P.json"))
        except Exception as e:
            logger.warning(role="rfecv_select", message=f"{tok}'s voting & config failed to save ({str(e).strip()})")
        else:
            logger.info(role="rfecv_select", message=f"{tok}'s voting & config saved")






