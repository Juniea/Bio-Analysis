from typing import Tuple

import os
import json
from pathlib import Path
from dotenv import load_dotenv

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

from tools import SyncLogger

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))  # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))  # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))  # path of the memory data directory


class Preprocessor:
    def __init__(
            self,
            beh_path: str | os.PathLike,
            sig_path: str | os.PathLike,
            tvt_ratio: Tuple[float, float, float] = (0.5, 0, 0.5),
            shuffle: bool = True,
            normalizer: str | None = 'z-score',
            agg: bool = False,
            smote: None | float | str = 'auto',
            resize: bool = False,
            target: str = 'cont',
            seed: int = 42,
            log_level: str = 'INFO'
    ) -> None:
        """
        The generic class for data preprocessing

        :param beh_path:    the path to the behavior file
        :param sig_path:    the path to the signal file
        :param tvt_ratio:   the ratio of (train_size : valid_size : test_size)
        :param shuffle:     The toggle for shuffling the dataset
        :param normalizer:  the type of normalizer to use
        :param agg:         the toggle for generate the SUM column (sum(all-signal-columns))
        :param smote:       the config for SMOTE
        :param resize:      the toggle for resizing when doing x-y alignment
        :param target:      the mode for labeling interested behaviors
        :param log_level:   the print level for logging
        """

        self.behavior_lookup = None
        self.behavior_df = pd.read_csv(beh_path)
        self.signal_df = pd.read_csv(sig_path)
        self.dataset_df = None
        self.core = {}

        self.beh_path = beh_path
        self.sig_path = sig_path
        self.shuffle = shuffle
        self.tvt_ratio = tvt_ratio
        self.normalizer = normalizer
        self.smote = smote
        self.agg = agg
        self.resize = resize
        self.target = target
        self.seed = seed

        self.logger = SyncLogger(level=log_level)
        self.sig_fps = 1 / (1.000045 if Path(sig_path).stem.split('_')[0] == 'K215' else 1.001286)
        self.beh_fps = 7.06
        self.beh_frame_to_second = lambda x: x / self.beh_fps
        self.beh_second_to_frame = lambda x: x * self.beh_fps
        self.sig_frame_to_second = lambda x: x / self.sig_fps
        self.sig_second_to_frame = lambda x: x * self.sig_fps

        # alignment
        self._config_target(mode=self.target)
        self._config_signal(deci=1)
        self._config_behavior(deci=1)
        self._align(deci=1)

        # transformation
        self._transform(
            tvt_ratio=tvt_ratio,
            shuffle=shuffle,
            smote_strategy=smote,
            normalizer=normalizer
        )

        # report the config
        self.logger.info(
            role="Predict.Preprocessor",
            message=f"Config: "
                    f"origin[{len(self.core['total_df_ori']) if self.core['total_df_ori'] is not None else '0'}] | "
                    f"train[{len(self.core['train_df_aft']) if self.core['train_df_aft'] is not None else '0'}] | "
                    f"valid[{len(self.core['valid_df_aft']) if self.core['valid_df_aft'] is not None else '0'}] | "
                    f"test[{len(self.core['test_df_aft']) if self.core['test_df_aft'] is not None else '0'}]"
        )

    def _config_random(self, seed: int = 42) -> None:
        self.logger.info(role="Predict.Preprocessor", message=f"Random seed set: {seed}")

        random.seed(seed)
        np.random.seed(seed)

    def _config_target(self, mode: str = "cont") -> None:
        """
        populate the "tag -> label" lookup table as self.behavior_lookup

        :param mode:  The mode for the lookup table setting

        :return: None
        """
        lookup = {
            # current interests
            'walk': 0,

            'cont reach': 0,
            'cont  reach': 0,
            'cont reaching': 0,
            'cont in vain': 0,
            'cont missed': 0,

            'bothreach': 0,
            'both reach': 0,
            'both paw reach': 0,
            'both reaching': 0,
            'both in vain': 0,
            'both invain': 0,
            'both paw in vain': 0,
            'both missed': 0,
            'bothmissed': 0,
            'cont miss': 0,

            # Mice Actions
            'tongue reach': 0,
            'tongue success': 0,
            'ipsi moved': 0,
            'ipsi in vain': 0,
            'both licking': 0,
            'cont licking': 0,
            'tongue': 0,
            'both paw licking': 0,
            'rt paw licking': 0,
            'cont paw movement': 0,
            'cont paw licking': 0,
            'cont lick': 0,
            'cont lift': 0,
            'right paw walk': 0,

            # 2-Photon Actions
            'last 2p': 0,
            'last 2p frame': 0,
            'last frame': 0,
            'noise generation': 0,
            'noist generation': 0,
            'spout movement': 0,
            'spout removed': 0,
            'valve opened': 0,
            'noise genearation': 0,
            'two photon imaging': 0,
            'two-photon imaging': 0,
            'spout removed after whisker detection': 0,
            'spout movement after whisker detection': 0,
            'spout moved after whisker detection': 0,
            'spout moved after wrong detection': 0,
        }

        if mode == 'both&inv':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['bothreach'] = 1
            self.behavior_lookup['both reach'] = 1
            self.behavior_lookup['both reaching'] = 1
            self.behavior_lookup['both in vain'] = 1
            self.behavior_lookup['both invain'] = 1
            self.behavior_lookup['both paw in vain'] = 1

        elif mode == 'cont&inv':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['cont reach'] = 1
            self.behavior_lookup['cont  reach'] = 1
            self.behavior_lookup['cont reaching'] = 1
            self.behavior_lookup['cont in vain'] = 1

        elif mode == 'both':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['bothreach'] = 1
            self.behavior_lookup['both reach'] = 1
            self.behavior_lookup['both reaching'] = 1

        elif mode == 'cont':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['cont reach'] = 1
            self.behavior_lookup['cont  reach'] = 1
            self.behavior_lookup['cont reaching'] = 1

        elif mode == 'walk':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['walk'] = 1

        elif mode == 'both_inv':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['both in vain'] = 1
            self.behavior_lookup['both invain'] = 1
            self.behavior_lookup['both paw in vain'] = 1

        elif mode == 'cont_inv':
            self.behavior_lookup = lookup.copy()
            self.behavior_lookup['cont in vain'] = 1

        else:
            self.logger.error(role="Predictor.Preprocessor",
                              message=f"unknown mode when configuring self.behavior_lookup: {mode}")

    def _config_signal(self, deci: int = 1) -> None:
        """
        populate the signal pd.Series as self.signal

        :param deci: the offset of the floating point that decides to what extent the rounded up should be for the time indexes

        :return: None
        """

        # Set the Index column for signal CSV
        if 'x0000' in self.signal_df.columns:
            self.signal_df = self.signal_df.set_index('x0000')
            self.signal_df = self.signal_df.rename_axis('index', axis='index')

        if self.agg:
            agg_sr = pd.Series(self.signal_df.sum(axis=1), name='SUM')
            self.signal_df = pd.concat([self.signal_df, agg_sr], axis=1)

        # Fill the gap at a certain frequency for Signal CSV
        core_index = np.array(self.signal_df.index.to_list(), dtype=np.float32)
        core_index = self.sig_frame_to_second(core_index)
        core_index = np.round(core_index, decimals=deci)

        fill_index = np.arange(core_index[0], core_index[-1] + 10 ** (-deci), 10 ** (-deci), dtype=np.float32)
        fill_index = np.round(fill_index, decimals=deci)

        self.signal_df.index = core_index
        self.signal_df = self.signal_df.reindex(fill_index)
        self.signal_df = self.signal_df.interpolate(axis=0, method='akima').ffill().bfill()

    def _config_behavior(self, deci: int = 1) -> None:
        """
        populate the behavior pd.Series as self.behavior

        :param deci: the offset of the floating point that decides to what extent the rounded up should be for the time indexes

        :return: None
        """

        self.behavior_df = self.behavior_df.loc[:, ['Behavior', 'Start', 'End']]

        # Extra 0.5 second offset (0.5 + 1.5[already added in the CSVs] = 2.0)
        self.behavior_df.loc[:, ['Start', 'End']] += self.beh_second_to_frame(0.5)

        # transform the frame numbers into the corresponding sec timestamps
        self.behavior_df.loc[:, ['Start', 'End']] = (
            self.behavior_df.loc[:, ['Start', 'End']]
            .map(self.beh_frame_to_second)
            .map(lambda x: np.round(x, decimals=deci))
        )

        # create the init labels pd.Series for populating the interested behaviors later
        start = self.behavior_df.loc[:, "Start"].iloc[0]
        end = self.behavior_df.loc[:, 'End'].iloc[-1]

        fill_index = np.arange(start, end + 10 ** (-deci), 10 ** (-deci), dtype=np.float32)
        fill_index = np.round(fill_index, decimals=deci)

        labels = pd.Series(np.zeros(len(fill_index)), index=fill_index, name='Label')

        # populating the interested behaviors labels
        for idx, row in self.behavior_df.iterrows():
            key = row.loc['Behavior'].strip().lower()
            if '(' in key:
                key = key[:-4].strip()

            if self.behavior_lookup[key] == 1:
                labels.loc[row.loc['Start']: row.loc['End']] = self.behavior_lookup[key]

        self.behavior_df = labels

    def _align(self, deci: int = 1) -> None:
        """
        Align the previously independent signal pd.Series and behavior pd.Series and
        do proper starting point alignment and cutting / shrinking / expanding if necessary.
        Populate self.dataset_df using the result

        :param deci:

        :return: None
        """

        # cut the extra part of the signal file which happens before the camera USB is triggered
        self.signal_df = self.signal_df.loc[self.behavior_df.index[0]:, :]

        # do the shrinking / expansion alignment
        if self.resize:
            # shrink the signal to match behavior
            if len(self.signal_df) > len(self.behavior_df):
                long_length = len(self.signal_df)
                short_length = len(self.behavior_df)

                # pick {short_length} rows from the original signal dataframe evenly
                idx = np.linspace(0, long_length - 1, short_length, dtype=int)
                self.signal_df = pd.DataFrame(
                    self.signal_df.iloc[idx, :].values,
                    index=self.behavior_df.index,
                    columns=self.signal_df.columns
                )

                self.behavior_df.index = self.behavior_df.index.map(lambda x: np.round(x, decimals=deci))
                self.signal_df.index = self.signal_df.index.map(lambda x: np.round(x, decimals=deci))

            # expand the signal to match behavior
            else:
                long_length = len(self.behavior_df)
                short_length = len(self.signal_df)
                ori_short = self.signal_df.values

                # create {long_length - short_length} extra rows from the original signal dataframe evenly
                idx_old = np.arange(short_length)
                idx_new = np.linspace(0, short_length - 1, long_length)
                new_short = np.apply_along_axis(
                    lambda col: np.interp(idx_new, idx_old, col),
                    axis=0,
                    arr=ori_short
                )
                self.signal_df = pd.DataFrame(new_short, index=self.behavior_df.index, columns=self.signal_df.columns)

                self.behavior_df.index = self.behavior_df.index.map(lambda x: np.round(x, decimals=deci))
                self.signal_df.index = self.signal_df.index.map(lambda x: np.round(x, decimals=deci))

        # do the cutting alignment
        else:
            # cut the extra part at the end of the signal to match behavior
            if len(self.signal_df) > len(self.behavior_df):
                self.signal_df = self.signal_df.iloc[:len(self.behavior_df), :]

            # cut the extra part at the end of the behavior to match signal
            else:
                self.behavior_df = self.behavior_df.iloc[:len(self.signal_df)]

        # assert equal length after alignment
        if len(self.signal_df) != len(self.behavior_df):
            self.logger.error(role="Predict.Preprocessor",
                              message="signal and behavior DataFrames do not match after alignment")

        # combine the aligned signal pd.Dataframe with the aligned behavior pd.Dataframe into the total dataset
        self.dataset_df = self.signal_df.copy()
        self.dataset_df.loc[:, self.behavior_df.name] = self.behavior_df.values

    def _transform(
            self,
            tvt_ratio: Tuple[float, float, float] = (0.5, 0.0, 0.5),
            shuffle: bool = True,
            smote_strategy: float | str | None = 'auto',
            normalizer: str = 'z-score',
    ) -> None:
        """
        Transform the dataset

        :param tvt_ratio:       The ratio of (train_size : valid_size : test_size)
        :param shuffle:         The toggle for shuffling the dataset (will happen before the train-validate-test-split)
        :param smote_strategy:  The smote config (None for disable, other please refer to https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
        :param normalizer:      The normalization method (will perform appropriately if tt_ratio != (1, 0, 0))

        :return: None
        """

        dataset = self.dataset_df.copy()

        # shuffle the dataset in the first place
        if shuffle:
            dataset = dataset.sample(frac=1, random_state=self.seed)

        # split the train, valid, test dataset immediately, preventing information leak
        if sum(tvt_ratio) > 1.0:
            self.logger.error(role="Predict.Preprocessor",
                              message=f"the tvt_ratio should be less or equal to 1.0, {sum(tvt_ratio)} instead")

        if tvt_ratio[0] <= 0:
            self.logger.error(role="Predict.Preprocessor",
                              message=f"train dataset size should be greater than 0, {tvt_ratio[0]} instead")

        if tvt_ratio[1] < 0 or tvt_ratio[2] < 0:
            self.logger.error(role="Predict.Preprocessor", message=f"tvt_ratio should be non-negative")

        train_len, valid_len, test_len = list(map(lambda x: int(x * len(dataset)), tvt_ratio))
        train_df = dataset.iloc[:train_len, :]
        valid_df = dataset.iloc[train_len:(train_len + valid_len), :] if valid_len else None
        test_df = dataset.iloc[(train_len + valid_len):, :] if test_len else None

        # SMOTE for train, valid, test separately therefore preventing information leak and shuffle immediately
        if smote_strategy is not None:
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=self.seed)
            train_x, train_y = smote.fit_resample(train_df.iloc[:, :-1], train_df.iloc[:, -1])
            train_df = pd.concat((train_x, train_y), axis=1)
            if shuffle:
                train_df = train_df.sample(frac=1, random_state=self.seed)

        total_df = pd.concat(list(filter(lambda x: x is not None, [train_df, valid_df, test_df])), axis=0)

        # normalization for all signal columns without information leak
        if normalizer == "z-score":
            ct = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler(), train_df.columns[:-1])
                ],
                remainder="passthrough"
            )
            ct.fit(train_df)
            total_df = pd.DataFrame(ct.transform(total_df), index=total_df.index, columns=total_df.columns)
            train_df = pd.DataFrame(ct.transform(train_df), index=train_df.index, columns=train_df.columns)
            valid_df = pd.DataFrame(
                ct.transform(valid_df),
                index=valid_df.index,
                columns=valid_df.columns
            ) if valid_df is not None else None
            test_df = pd.DataFrame(
                ct.transform(test_df),
                index=test_df.index,
                columns=test_df.columns
            ) if test_df is not None else None
            self.dataset_df = pd.DataFrame(
                ct.transform(self.dataset_df),
                index=self.dataset_df.index,
                columns=self.dataset_df.columns
            )

        elif normalizer == "min-max":
            ct = ColumnTransformer(
                transformers=[
                    ("scaler", MinMaxScaler(), train_df.columns[:-1])
                ],
                remainder="passthrough"
            )
            ct.fit(train_df)

            total_df = pd.DataFrame(ct.transform(total_df), index=total_df.index, columns=total_df.columns)
            train_df = pd.DataFrame(ct.transform(train_df), index=train_df.index, columns=train_df.columns)
            valid_df = pd.DataFrame(
                ct.transform(valid_df),
                index=valid_df.index,
                columns=valid_df.columns
            ) if valid_df is not None else None
            test_df = pd.DataFrame(
                ct.transform(test_df),
                index=test_df.index,
                columns=test_df.columns
            ) if test_df is not None else None
            self.dataset_df = pd.DataFrame(
                ct.transform(self.dataset_df),
                index=self.dataset_df.index,
                columns=self.dataset_df.columns
            )

        self.core["total_df_ori"] = self.dataset_df.copy()
        self.core["total_df_aft"] = total_df
        self.core["train_df_aft"] = train_df
        self.core["valid_df_aft"] = valid_df
        self.core["test_df_aft"] = test_df

    def get_origin_total(self, shard="all"):
        if shard == "all":
            return self.core["total_df_ori"]
        elif shard == "x":
            return self.core["total_df_ori"].iloc[:, :-1]
        elif shard == "y":
            return self.core["total_df_ori"].iloc[:, -1]
        else:
            self.logger.error(role="Predict.Preprocessor", message=f"Unknown shard: {shard}")
            return None

    def get_after_total(self, shard="all"):
        if shard == "all":
            return self.core["total_df_aft"]
        elif shard == "x":
            return self.core["total_df_aft"].iloc[:, :-1]
        elif shard == "y":
            return self.core["total_df_aft"].iloc[:, -1]
        else:
            self.logger.error(role="Predict.Preprocessor", message=f"Unknown shard: {shard}")
            return None

    def get_train(self, shard="all"):
        if shard == "all":
            return self.core["train_df_aft"]
        elif shard == "x":
            return self.core["train_df_aft"].iloc[:, :-1]
        elif shard == "y":
            return self.core["train_df_aft"].iloc[:, -1]
        else:
            self.logger.error(role="Predict.Preprocessor", message=f"Unknown shard: {shard}")
            return None

    def get_valid(self, shard="all"):
        if shard == "all":
            return self.core["valid_df_aft"] if self.core["valid_df_aft"] is not None else None
        elif shard == "x":
            return self.core["valid_df_aft"].iloc[:, :-1] if self.core["valid_df_aft"] is not None else None
        elif shard == "y":
            return self.core["valid_df_aft"].iloc[:, -1] if self.core["valid_df_aft"] is not None else None
        else:
            self.logger.error(role="Predict.Preprocessor", message=f"Unknown shard: {shard}")
            return None

    def get_test(self, shard="all"):
        if shard == "all":
            return self.core["test_df_aft"] if self.core["valid_df_aft"] is not None else None
        elif shard == "x":
            return self.core["test_df_aft"].iloc[:, :-1] if self.core["test_df_aft"] is not None else None
        elif shard == "y":
            return self.core["test_df_aft"].iloc[:, -1] if self.core["test_df_aft"] is not None else None
        else:
            self.logger.error(role="Predict.Preprocessor", message=f"Unknown shard: {shard}")
            return None

    def to_dict(self):
        config = {
            "beh_path": str(self.beh_path),
            "sig_path": str(self.sig_path),
            "tvt_ratio": self.tvt_ratio,
            "shuffle": self.shuffle,
            "normalizer": self.normalizer,
            "agg": self.agg,
            "smote": self.smote,
            "resize": self.resize,
            "target": self.target,
            "seed": self.seed,
        }

        return config

    def json(self, path: str | os.PathLike | Path) -> None:
        with path.open("w") as fp:
            json.dump(self.to_dict(), fp)

        self.logger.debug(role="Preprocessor", message=f"preprocessor object saved to {path}")


# Using Sample
if __name__ == '__main__':
    token = "K228_B_D8"
    preprocessor = Preprocessor(
        beh_path=PATH_DATA / Path(f'{token}_B.csv'),
        sig_path=PATH_DATA / Path(f'{token}_S.csv'),
        tvt_ratio=(0.5, 0, 0.5),
        shuffle=True,
        normalizer='min-max',
        agg=True,
        smote='auto',
        resize=True,
        target="cont",
        log_level="DEBUG"
    )

    dataset = preprocessor.get_origin_total(shard="all")
    dataset.iloc[:, 0].plot(
        kind="line",
        lw=0.9,
        color='royalblue',
        title='signal along time',
        xlabel='time',
        ylabel='intensity',
        figsize=(16, 9),
    )
    plt.show()
    plt.close()