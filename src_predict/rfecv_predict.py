import os
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import ConfusionMatrixDisplay
import lightgbm as lgb

from tools import SyncLogger, check_folders, fetch_tokens
from .preprocessor import Preprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


@check_folders(
    "sel_folder",
    "src_folder",
    "tgt_int_folder",
    "tgt_mat_folder",
    "tgt_raw_folder"
)
def rfecv_predict(
    sel_folder: str | os.PathLike | Path,
    src_folder: str | os.PathLike | Path,
    tgt_int_folder: str | os.PathLike | Path,
    tgt_mat_folder: str | os.PathLike | Path,
    tgt_raw_folder: str | os.PathLike | Path,
    beh: str                = "cont",
    on_predict: Callable    = lambda x: True,
    standard: int           = 3,
    skew: bool              = False,
    seed: int               = 42,
    axis: bool              = True,
    colormap: str           = "Reds",
    rerun: bool             = False,
) -> None:
    """
    Predicting the behaviors using the training set only

    :param sel_folder: The folder that contains the RFECV-SELECT-ed cells
    :param src_folder: The folder that contains the entire dataset
    :param tgt_int_folder: The folder where the prediction intervals will be placed
    :param tgt_mat_folder: The folder where the prediction confusion matrices will be placed
    :param tgt_raw_folder: The folder where the prediction labels will be placed
    :param beh: The interested behaviors
    :param on_predict: The extra filtering function so that only the interested tokens will be predicted
    :param standard: The least number for choosing a RFECV-SELECT-ed cell
    :param skew: The toggle for shuffling the cell-signal features on the time axis
    :param seed: The random seed for the prediction model and the `skew` shuffling
    :param axis: The toggle for adding the axis on the prediction interval plots
    :param colormap: The color map to-be-applied on the prediction interval plots and the confusion matrices
    :param rerun: The toggle for rerunning the prediction entirely

    :return: None
    """

    logger = SyncLogger(level="INFO")

    # generate all paths for data files to-be-predict
    tokens = fetch_tokens(folder=src_folder, shard="all")
    tokens = tuple(filter(on_predict, tokens))

    # iterate through all tokens for prediction
    for token in tokens:
        if (
                (not rerun) and
                (Path(tgt_int_folder) / Path(f'{token}-{beh}.svg')).exists() and
                (Path(tgt_mat_folder) / Path(f'{token}-{beh}.pdf')).exists() and
                (Path(tgt_raw_folder) / Path(f'{token}-{beh}.csv')).exists()
        ):
            logger.info(role="rfecv_predict", message=f"{token} already computed")
            continue

        # reproduce the preprocessor config when the RFECV-SELECT happened
        try:
            with open(Path(sel_folder) / Path(f'{token}_P.json'), 'r') as fp:
                prep_config = json.load(fp)
        except FileNotFoundError as e:
            logger.warning(role="rfecv_predict", message=f"{token} failed to load due to {str(e)}")
            continue

        try:
            prep = Preprocessor(**prep_config, log_level="WARNING")
        except Exception as e:
            logger.warning(role="rfecv_predict", message=f"{token} failed to load due to {str(e)}")
            continue

        # fetching the features that has votes greater than the standard and generate the datasets
        vote = pd.read_csv(Path(sel_folder) / Path(f"{token}_V.csv"))
        vote = vote.set_index('features')
        features = vote.loc[vote.sum(axis=1) >= standard].index.to_list()

        total_x = prep.get_origin_total(shard="x").loc[:, features]
        total_y = prep.get_origin_total(shard="y").values
        train_x = prep.get_train(shard="x").loc[:, features]
        train_y = prep.get_train(shard="y").values
        test_x = prep.get_test(shard="x").loc[:, features]
        test_y = prep.get_test(shard="y").values

        # generate control group for RFECV-PREDICT
        if skew:
            np.random.seed(seed)
            train_x = train_x.apply(np.random.permutation)

        # model fitting on the training dataset and predict on the total & testing dataset
        model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            num_leaves=31,
            verbose=-1,
            n_jobs=-1,
            random_state=seed
        )
        model.fit(train_x, train_y)
        total_p = model.predict(total_x)
        test_p = model.predict(test_x)

        # select the proper colormap for plotting
        if colormap.strip().lower() == "blues":
            cmap_interval = ListedColormap(('lightsteelblue', '#0e56aa'))
            cmap_matrix = "Blues"
        elif colormap.strip().lower() == "reds":
            cmap_interval = ListedColormap(("#fbddc6", "#c43639"))
            cmap_matrix = "Reds"
        else:
            cmap_interval = ListedColormap(("white", "black"))
            cmap_matrix = "gist_yarg"

        plt.figure(figsize=(16, 12))

        # plotting for prediction interval
        ax = plt.subplot(2, 1, 1)
        if not axis:
            ax.set(xticks=[], yticks=[])
            plt.title("")
        else:
            ax.set(yticks=[])
            plt.title("Prediction")
        ax.imshow(total_p.reshape((1, -1)), cmap=cmap_interval, aspect="auto")

        # plotting for true interval
        ax = plt.subplot(2, 1, 2)
        if not axis:
            ax.set(xticks=[], yticks=[])
            plt.title("")
        else:
            ax.set(yticks=[])
            plt.title("Real")
        ax.imshow(total_y.reshape((1, -1)), cmap=cmap_interval, aspect="auto")

        # interval plots saving
        plt.savefig(Path(tgt_int_folder) / Path(f"{token}-{beh}.svg"))
        plt.close()

        # plotting for confusion matrix (for testing dataset only)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            test_y,
            test_p,
            display_labels=['None', beh],
            cmap=cmap_matrix,
            im_kw={'vmin': 0.0, 'vmax': 1.0},
            normalize='true',
            ax=ax
        )

        if not axis:
            ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])

        # confusion matrix saving
        plt.savefig(Path(tgt_mat_folder) / Path(f"{token}-{beh}.pdf"))
        plt.close()

        # raw data saving
        pd.DataFrame(
            np.stack([total_p, total_y], axis=1), columns=["prediction", "real"]
        ).to_csv(Path(tgt_raw_folder) / Path(f"{token}-{beh}.csv"), index=False)

        logger.info(role="rfecv_predict", message=f"{token} interval & matrix & record saved")



if __name__ == '__main__':
    rfecv_predict(
        sel_folder=PATH_CACHE / Path("rfecv_init_vote/cont"),
        src_folder=PATH_DATA,
        tgt_int_folder=PATH_PLOT / Path(f"rfecv_predict/interval_skew"),
        tgt_mat_folder=PATH_PLOT / Path(f"rfecv_predict/matrix_skew"),
        tgt_raw_folder=PATH_PLOT / Path(f"rfecv_predict/raw_skew"),
        beh="cont",
        standard=3,
        seed=42,
        skew=True,
        axis=False,
        colormap="reds",
        rerun=True
    )