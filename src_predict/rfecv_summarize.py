import os
from pathlib import Path
from typing import Callable, Tuple, Dict

from dotenv import load_dotenv

import json
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

import lightgbm as lgb
import xgboost as xgb

from tools import SyncLogger, check_folders, fetch_tokens
from .preprocessor import Preprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


class Summarizer:
    @check_folders("src_folder")
    def __init__(
        self,
        src_folder: str | os.PathLike | Path,
        seed: int = 42,
    ) -> None:
        """

        :param src_folder:
        :param seed:
        """

        self.seed = seed
        self.src_folder = Path(src_folder)

        self.logger = SyncLogger(level="INFO")
        self.model_name_lookup = {
            'LR': 'Lasso',
            'SVM': 'SVM',
            'RF': 'RandomForest',
            'XGB': 'XGBoost',
            'LGB': 'LightGBM'
        }
        self.model_lookup = {
            'LR': LogisticRegression(
                n_jobs=4,
                max_iter=50000,
                random_state=self.seed
            ),
            'SVM': LinearSVC(
                penalty='l1',
                max_iter=50000,
                random_state=self.seed,
            ),
            'RF': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=self.seed
            ),
            'ADA': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=self.seed),
                n_estimators=100,
                random_state=self.seed
            ),
            'LGB': lgb.LGBMClassifier(
                boosting_type='gbdt',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                num_leaves=31,
                verbose=-1,
                n_jobs=-1,
                random_state=self.seed
            ),
            'XGB': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=self.seed,
            ),
        }

    @check_folders("sel_folder", "tgt_folder")
    def get_scores(
        self,
        sel_folder: str | os.PathLike | Path,
        tgt_folder: str | os.PathLike | Path,
        beh: str                = "cont",
        on_predict: Callable    = lambda x: True,
        models: Tuple[str, ...] = ('LR', 'SVM', 'RF', 'XGB', 'LGB'),
        standard: int           = 3,
        rerun: bool             = False,
    ) -> Dict:
        """


        :param sel_folder:
        :param tgt_folder:
        :param beh:
        :param on_predict:
        :param models:
        :param standard:
        :param rerun:

        :return:
        """

        score_lookup = {}
        modified = False

        # check if the scores are already cached
        if not (Path(sel_folder) / Path(f"scoring-{beh}.json")).exists():
            rerun = True
        else:
            with open(Path(sel_folder) / Path(f"scoring-{beh}.json")) as fp:
                score_lookup = json.load(fp)

        # iterate through all tokens for score computation
        tokens = fetch_tokens(folder=self.src_folder, shard="all")
        tokens = tuple(filter(on_predict, tokens))

        # iterate through all feature groups
        for feature_key in ("sel_features", "oth_features", "all_features"):

            if feature_key not in score_lookup.keys():
                score_lookup[feature_key] = {}

            # iterate through all models
            for model_name in models:

                if model_name not in score_lookup[feature_key].keys():
                    score_lookup[feature_key][model_name] = {}

                # iterate through all tokens
                for token in tokens:
                    # skip already-computed `feature_key-model_name-token` entities
                    if (
                            (not rerun) and
                            score_lookup.get(feature_key, False) and
                            score_lookup[feature_key].get(model_name, False) and
                            score_lookup[feature_key][model_name].get(token, False)
                    ):
                        self.logger.info(role="Summarizer.get_scores", message=f"{token} scores already computed")
                        continue

                    # reproduce the preprocessor config when the RFECV-SELECT happened
                    try:
                        with open(Path(sel_folder) / Path(f'{token}_P.json'), 'r') as fp:
                            prep_config = json.load(fp)
                    except FileNotFoundError as e:
                        self.logger.warning(role="rfecv_predict", message=f"{token} failed to load due to {str(e)}")
                        continue

                    try:
                        prep = Preprocessor(**prep_config, log_level="WARNING")
                    except Exception as e:
                        self.logger.warning(role="rfecv_predict", message=f"{token} failed to load due to {str(e)}")
                        continue

                    vote = pd.read_csv(Path(sel_folder) / Path(f"{token}_V.csv"))
                    vote = vote.set_index('features')

                    # fetch the features according to the current feature group
                    if feature_key == "sel_features":
                        features = vote.loc[vote.sum(axis=1) >= standard].index.to_list()
                    elif feature_key == "oth_features":
                        features = vote.loc[vote.sum(axis=1) <  standard].index.to_list()
                    elif feature_key == "all_features":
                        features = vote.index.to_list()
                    else:
                        self.logger.error(role="get_scores", message=f"invalid feature_key: {feature_key}")
                        raise RuntimeError(f"invalid feature_key: {feature_key}")

                    if len(features) <= 0:
                        self.logger.warning(role="get_scores", message=f"skip model fitting due to empty features: {token}-{feature_key}")
                        continue

                    # assemble the train & test datasets according to the `feature_key-token` pair
                    train_x = prep.get_train(shard="x").loc[:, features]
                    train_y = prep.get_train(shard="y").values
                    test_x =  prep.get_test(shard="x").loc[:, features]
                    test_y =  prep.get_test(shard="y").values

                    # predict using the current model_name
                    model = deepcopy(self.model_lookup[model_name])
                    model.fit(train_x, train_y)
                    test_p = model.predict(test_x)
                    score = f1_score(test_y, test_p)

                    score_lookup[feature_key][model_name][token] = score
                    modified = True

        # save the scoring lookup for later caching
        if modified:
            with open(Path(tgt_folder) / Path('scoring.json'), 'w') as fp:
                json.dump(score_lookup, fp, indent=4)

        self.logger.info(role="Summarizer.get_scores", message=f"{beh} scores computed")

        return score_lookup

    @check_folders("sel_folder", "cac_folder", "tgt_folder")
    def display_bar(
        self,
        sel_folder: str | os.PathLike | Path,
        cac_folder: str | os.PathLike | Path,
        tgt_folder: str | os.PathLike | Path,
        beh: str                = "cont",
        on_predict: Callable    = lambda x: True,
        models: Tuple[str, ...] = ('LR', 'SVM', 'RF', 'XGB', 'LGB'),
        standard: int           = 3,
        rerun: bool             = False,
    ) -> None:
        """

        :param sel_folder:
        :param cac_folder:
        :param tgt_folder:
        :param beh:
        :param on_predict:
        :param models:
        :param standard:
        :param rerun:

        :return:
        """

        # compute all scores for the given behavior
        score_lookup = self.get_scores(
            sel_folder=sel_folder,
            tgt_folder=cac_folder,
            beh="cont",
            on_predict=on_predict,
            models=models,
            standard=standard,
            rerun=rerun,
        )

        # compute all avg scores for each `feature_key-model_key` pair
        avg_score_lookup = {}
        for feature_key, model_dict in score_lookup.items():
            if avg_score_lookup.get(feature_key, None) is None:
                avg_score_lookup[feature_key] = {}

            for model_key, token_dict in model_dict.items():
                if avg_score_lookup[feature_key].get(model_key, None) is None:
                    avg_score_lookup[feature_key][model_key] = {}

                avg_score_lookup[feature_key][model_key] = np.array(list(token_dict.values())).mean()

        # plot the bar plot for scoring displaying
        fig, ax = plt.subplots(figsize=(16, 9))

        width = 0.3
        x = np.arange(len(models))

        ax.bar(
            x - width,
            list(avg_score_lookup['sel_features'].values()),
            width,
            label='RFECV',
            color="#56a4d9",
        )

        ax.bar(
            x,
            list(avg_score_lookup['oth_features'].values()),
            width,
            label='Other',
            color="#4e7abd",
        )

        ax.bar(
            x + width,
            list(avg_score_lookup['all_features'].values()),
            width,
            label='ALL',
            color="#d1d1d1",
        )

        plt.title(f'F1-score average for predicting {beh}')
        plt.xlabel('Models')
        plt.ylabel('Average F1-scores')

        ax.set_xticks(x)
        ax.set_ylim((0.0, 1.0))
        ax.set_xticklabels([self.model_name_lookup[model_tag] for model_tag in models])

        plt.legend(loc="upper right")
        plt.tight_layout(pad=1.0)

        # save the prediction bar plot
        plt.savefig(tgt_folder / Path(f"prediction_bar-{beh}.pdf"))
        plt.show()
        self.logger.info(role="Summarizer", message=f"{beh} prediction bars plotted")

        plt.close()

    @check_folders("sel_folder", "tgt_folder")
    def display_pie(
        self,
        sel_folder: str | os.PathLike | Path,
        tgt_folder: str | os.PathLike | Path,
        beh: str                = 'cont',
        on_predict: Callable    = lambda x: True,
        standard: int           = 4,
    ) -> None:
        """

        :param sel_folder:
        :param tgt_folder:
        :param beh:
        :param on_predict:
        :param standard:

        :return: None
        """

        # get the len(sel_features) & len(all_features) for a given token
        def get_nums_unit(token):
            try:
                votes = pd.read_csv(Path(sel_folder) / Path(f"{token}_V.csv"))
            except FileNotFoundError as e:
                self.logger.warning(role="Summarizer", message=f"failed to add {token} into account as its voting file is not found")
                return 0, 0
            else:
                votes = votes.set_index('features')
                features = votes.loc[votes.sum(axis=1) >= standard].index.to_list()

            return len(features), len(votes)

        # iterate through all tokens for score computation
        tokens = fetch_tokens(folder=self.src_folder, shard="all")
        tokens = tuple(filter(on_predict, tokens))

        sum_select_num = 0
        sum_all_num = 0
        for token in tokens:
            select_num, all_num = get_nums_unit(token)
            sum_select_num += select_num
            sum_all_num += all_num

        # plot the bar plot for occupation displaying
        sizes = [sum_select_num, sum_all_num]
        labels = ['RFECV', 'Other']
        explode = [0.05, 0]
        colors = ["#56a4d9", "#d1d1d1"]

        plt.figure(figsize=(12, 9))

        plt.pie(
            sizes,
            labels=labels,
            explode=explode,
            autopct='%1.2f%%',
            startangle=90,
            colors=colors,
            textprops=dict(
                color='white',
                fontsize=16,
                fontweight='bold'
            )
        )

        plt.axis('equal')
        plt.title(f'RFECV vs. Other with standard {standard}')

        plt.tight_layout(pad=1.0)
        plt.legend(loc="upper right")

        # save the prediction pie plot
        plt.savefig(Path(tgt_folder) / Path(f"prediction_pie-{beh}.pdf"))
        plt.show()
        self.logger.info(role="Summarizer", message=f"{beh} prediction pies plotted")

        plt.close()


if __name__ == '__main__':
    # Draw the distribution for all mice
    sm = Summarizer(
        seed=42,
        src_folder=PATH_DATA
    )

    # sm.display_bar(
    #     sel_folder=PATH_CACHE / Path("rfecv_init_vote/cont"),
    #     cac_folder=PATH_CACHE / Path("rfecv_init_scor/cont"),
    #     tgt_folder=PATH_PLOT  / Path("rfecv_predict/summary"),
    #     beh="cont",
    #     on_predict=lambda x: True,
    #     models=('LR', 'SVM', 'RF', 'XGB', 'LGB'),
    #     standard=3,
    #     rerun=True,
    # )

    # sm.display_pie(
    #     sel_folder=PATH_CACHE / Path("rfecv_init_vote/cont"),
    #     tgt_folder=PATH_PLOT / Path("rfecv_predict/summary"),
    #     beh="cont",
    #     on_predict=lambda x: True,
    #     standard=3,
    # )