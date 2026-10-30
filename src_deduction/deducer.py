from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple, Self, cast
from functools import partial
from copy import deepcopy

from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import multiprocessing

import os
import json
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tools import SyncLogger, check_folders, fetch_tokens
from .preprocessor import Preprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


# The dataclass describing a locator's output
@dataclass
class Window:
    token:          None | str
    beh:            None | str
    curve:          None | pd.DataFrame
    label:          None | pd.Series
    spike_status:   None | pd.DataFrame
    label_status:   None | pd.DataFrame
    valid:          None | bool

    def __repr__(self):
        return f"Window(token={self.token}, len4window={len(self.label_status)}, len4data={len(self.label)})"

    @staticmethod
    def json_load(path: str | os.PathLike | Path) -> "Window":
        # load the json checkpoint
        try:
            with open(Path(path), 'r') as fp:
                config = json.load(fp)

        except Exception as e:
            raise e

        else:
            config["curve"] = pd.DataFrame(
                config.get("curve", None),
                index=config.get("_curve_idx", None),
                columns=config.get("_curve_col", None)
            )
            config["label"] = pd.Series(
                config.get("label", None),
                index=config.get("_label_idx", None)
            )
            config["spike_status"] = pd.DataFrame(
                config.get("spike_status", None),
                index=config.get("_spike_status_idx", None),
                columns=config.get("_spike_status_col", None)
            )
            config["label_status"] = pd.DataFrame(
                config.get("label_status", None),
                index=config.get("_label_status_idx", None),
                columns=config.get("_label_status_col", None)
            )

            for key in list(config.keys()):
                if key.startswith("_"):
                    del config[key]

            return Window(**config)

    def json_dump(self, path: str | os.PathLike | Path) -> None:
        # transform pandas objects into pure lists
        ref_config = deepcopy(asdict(self))
        ans_config = deepcopy(ref_config)
        for key, value in ref_config.items():
            if isinstance(value, pd.Series):
                ans_config[key] = value.tolist()
                ans_config[f"_{key}_idx"] = value.index.tolist()

            elif isinstance(value, pd.DataFrame):
                ans_config[key] = value.values.tolist()
                ans_config[f"_{key}_idx"] = value.index.tolist()
                ans_config[f"_{key}_col"] = value.columns.tolist()

        # save the transformed config dict
        with open(Path(path), 'w') as fp:
            json.dump(ans_config, fp)


class Locator:
    @check_folders("sig_folder", 'beh_folder')
    def __init__(
        self,
        sig_folder: str | os.PathLike | Path,
        beh_folder: str | os.PathLike | Path,
        log_level: str = "INFO",
    ) -> None:
        self.sig_folder = Path(sig_folder)
        self.beh_folder = Path(beh_folder)

        self.logger = SyncLogger(level=log_level)

    @staticmethod
    def _get_sci(window, col_name):
        return window.spike_status.columns.get_loc(col_name)

    @staticmethod
    def _get_lci(window, col_name):
        return window.label_status.columns.get_loc(col_name)

    @staticmethod
    def _get_slices_on(
        funct: Callable,
        array: list
    ) -> List[List[int]]:
        """
        Get the list of all longest successive indexes that satisfy certain conditions

        :param funct: The condition function that takes in an element from the `array` and judge its property
        :param array: The list of elements

        :return: The list of indices that satisfy certain conditions
        """

        # Preparing window object
        left, right = 0, 0
        windows = []

        # Expanding the window
        while right < len(array):
            ele_right = array[right]
            right += 1

            while left < right and (not funct(ele_right) or right == len(array)):
                if funct(array[left]):
                    if right == len(array) and funct(ele_right):
                        windows.append(list(range(left, right)))
                    else:
                        windows.append(list(range(left, right - 1)))

                left = right

        return windows

    @check_folders("tgt_folder")
    def get_windows(
        self,
        token: str,
        beh: str = "cont",
        smooth: int = 1,
        threshold: float = 0.55,
        method: str = 'max',
        step: int = 1,
        rerun: bool = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> Dict[str, Window]:
        """

        :param token:
        :param beh:
        :param smooth:
        :param threshold:
        :param method:
        :param step:
        :param rerun:
        :param tgt_folder:

        :return:
        """

        try:
            if rerun:
                raise Exception("rerun hack")
            window_lookup = {
                "none":     Window.json_load(path=Path(tgt_folder) / Path(f"{token}_N.json")),
                "both":     Window.json_load(path=Path(tgt_folder) / Path(f"{token}_B.json")),
                "left":     Window.json_load(path=Path(tgt_folder) / Path(f"{token}_L.json")),
                "right":    Window.json_load(path=Path(tgt_folder) / Path(f"{token}_R.json"))
            }

        except Exception as e:
            # get all normalized aggregated signal & label index time series
            prep = Preprocessor(
                beh_path=self.sig_folder / Path(f"{token}_B.csv"),
                sig_path=self.beh_folder / Path(f"{token}_S.csv"),
                target=beh,
                tvt_ratio=(1.0, 0.0, 0.0),
                shuffle=False,
                normalizer='min-max',
                agg=True,
                smote=None,
                resize=True,
                log_level="WARNING"
            )

            # init the class-level global window
            ori_curve = prep.get_origin_total(shard='x')

            if smooth >= 1:
                # the execution function for smoothing
                def _smooth(series: pd.Series) -> pd.Series:
                    ori_series = series.to_list()
                    for idx, val in enumerate(series):
                        series.iloc[idx] = np.mean(ori_series[max(0, idx - smooth): min(len(ori_series) - 1, idx + smooth + 1)]).item()
                    return series

                results = Parallel(n_jobs=multiprocessing.cpu_count())(
                    delayed(_smooth)(ori_curve.loc[:, col])
                    for col in ori_curve.columns
                )

                aft_curve = pd.concat(results, axis=1)
                aft_curve.columns = ori_curve.columns
            else:
                aft_curve = ori_curve

            init_spike_status = self._init_spike(
                curve=aft_curve.loc[:, 'SUM'].tolist(),
                threshold=threshold,
                method=method,
            )

            window_lookup = {}
            window_lookup['none'] = Window(
                token=token,
                beh=beh,
                curve=aft_curve,
                label=prep.get_origin_total(shard='y'),
                spike_status=pd.DataFrame(
                    {
                        'left':     init_spike_status,
                        'center':   init_spike_status,
                        'right':    init_spike_status,
                        'hit':      False,
                    }
                ),
                label_status=pd.DataFrame(
                    {
                        'index':    np.where(np.array(prep.get_origin_total(shard="y")) == 1)[0].tolist(),
                        'hit':      False,
                    }
                ),
                valid=(np.array(prep.get_origin_total(shard='y')) == 1).any().item()
            )

            # extend the boarders for three directions in parallel
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self._push_spike, deepcopy(window_lookup['none']), step, "both"),
                    executor.submit(self._push_spike, deepcopy(window_lookup['none']), step, "left"),
                    executor.submit(self._push_spike, deepcopy(window_lookup['none']), step, "right"),
                ]

                window_lookup['both'], window_lookup['left'], window_lookup['right'] = [f.result() for f in futures]

                window_lookup['none' ].json_dump(path=Path(tgt_folder) / Path(f"{token}_N.json"))
                window_lookup['both' ].json_dump(path=Path(tgt_folder) / Path(f"{token}_B.json"))
                window_lookup['left' ].json_dump(path=Path(tgt_folder) / Path(f"{token}_L.json"))
                window_lookup['right'].json_dump(path=Path(tgt_folder) / Path(f"{token}_R.json"))

        self.logger.info(role="Locator", message=f"{token} window expanding completed")

        return window_lookup

    def _init_spike(
        self,
        curve: list,
        threshold=0.55,
        method='max'
    ) -> List[Dict]:
        """
        Get the description dict for peak locations

        :param threshold: The threshold to consider a peak
        :param method: The method to identify peaks

        :return: the description dict for peak locations
        """

        bar = abs(max(curve) - min(curve)) * threshold + min(curve)
        plateaus = self._get_slices_on(funct=lambda x: x >= bar, array=curve)

        spike_centers = []
        if method == 'max':
            for plateau in plateaus:
                chosen = plateau[0]
                for idx in plateau:
                    if curve[idx] > curve[chosen]:
                        chosen = idx
                spike_centers.append(chosen)

        elif method == 'center':
            spike_centers = list(
                map(
                    lambda x: x[len(x) // 2],
                    plateaus
                )
            )

        else:
            self.logger.error(role="Locator", message="unknown method for identifying the peaks")
            raise RuntimeError("unknown method for identifying the peaks")

        self.logger.debug(role="Locator", message="Done identifying the peaks")

        return spike_centers

    def _step_spike(
        self,
        window: Window,
        step: int = 1,
        direction: str = 'both',
    ) -> None:

        get_sci = partial(self._get_sci, window)
        length = len(window.spike_status)

        if direction == 'both':
            # expand all boarders for 1 step
            window.spike_status.iloc[0, get_sci('left')]  -= step
            window.spike_status.iloc[0, get_sci('right')] += step
            window.spike_status.iloc[0, get_sci('left')]   = max(
                window.spike_status.iloc[0, get_sci('left')],
                0
            )

            window.spike_status.iloc[-1, get_sci('left')]  -= step
            window.spike_status.iloc[-1, get_sci('right')] += step
            window.spike_status.iloc[-1, get_sci('right')]  = min(
                window.spike_status.iloc[-1, get_sci('right')],
                len(window.curve) - 1
            )

            for idx in range(1, length - 1):
                window.spike_status.iloc[idx, get_sci('left')]  -= step
                window.spike_status.iloc[idx, get_sci('right')] += step

            # merge all neighboring overlapping boarders
            for prev_idx, curr_idx in zip(range(0, length - 1), range(1, length)):
                prev_val = window.spike_status.iloc[prev_idx, :]
                curr_val = window.spike_status.iloc[curr_idx, :]

                if prev_val.loc['right'] > curr_val.loc['left']:
                    midd = (prev_val.loc['right'] + curr_val.loc['left']) // 2
                    window.spike_status.iloc[prev_idx, get_sci('right')] = midd
                    window.spike_status.iloc[curr_idx, get_sci('left')]  = midd


        elif direction == 'right':
            # expand all boarders for 1 step
            window.spike_status.iloc[-1, get_sci('right')] += step
            window.spike_status.iloc[-1, get_sci('right')]  = min(
                window.spike_status.iloc[-1, get_sci('right')],
                len(window.curve) - 1
            )

            for idx in range(0, length - 1):
                window.spike_status.iloc[idx, get_sci('right')] += step

            # merge all neighboring overlapping boarders
            for prev_idx, curr_idx in zip(range(0, length - 1), range(1, length)):
                prev_val = window.spike_status.iloc[prev_idx, :]
                curr_val = window.spike_status.iloc[curr_idx, :]

                if prev_val.loc['right'] > curr_val.loc['center']:
                    window.spike_status.iloc[prev_idx, get_sci('right')] = curr_val.loc['center']


        elif direction == 'left':
            # expand all boarders for 1 step
            window.spike_status.iloc[0, get_sci('left')]  -= step
            window.spike_status.iloc[0, get_sci('left')] = max(
                window.spike_status.iloc[0, get_sci('left')],
                0
            )

            for idx in range(1, length):
                window.spike_status.iloc[idx, get_sci('left')] -= step

            # merge all neighboring overlapping boarders
            for prev_idx, curr_idx in zip(range(0, length - 1), range(1, length)):
                prev_val = window.spike_status.iloc[prev_idx, :]
                curr_val = window.spike_status.iloc[curr_idx, :]

                if prev_val.loc['center'] > curr_val.loc['left']:
                    window.spike_status.iloc[curr_idx, get_sci('left')] = prev_val.loc['center']

    def _step_label(
        self,
        window: Window,
    ) -> None:
        """
        Update all hitting conditions for both the running peak configs and the running label configs

        :param window

        :return: None
        """

        get_sci = partial(self._get_sci, window)
        get_lci = partial(self._get_lci, window)

        for spike_idx, spike_status in enumerate(window.spike_status.itertuples()):
            # checking for inclusion conditions
            for label_idx, label_status in enumerate(window.label_status.itertuples()):
                if spike_status.left <= label_status.index <= spike_status.right:
                    window.spike_status.iloc[spike_idx, get_sci('hit')] = True
                    window.label_status.iloc[label_idx, get_lci('hit')] = True
                    break

    def _is_terminate(
        self,
        window: Window,
        direction: str = "both"
    ) -> bool:

        get_sci = partial(self._get_sci, window)

        # check if all labels are hit
        if window.label_status.loc[:, 'hit'].all() or not window.valid:
            return True

        # check if all neighboring boarders are merged
        if direction == 'both':
            if (
                    window.spike_status.iloc[0, get_sci('left')] > 0 or
                    window.spike_status.iloc[-1, get_sci('right')] < len(window.curve) - 1
            ):
                return False

            return (window.spike_status.loc[:, "right"].shift(0) == window.spike_status.loc[:, "left"].shift(-1)).iloc[:-1].all()

        elif direction == 'left':
            if window.spike_status.iloc[0, get_sci('left')] > 0:
                return False

            return (window.spike_status.loc[:, "center"].shift(0) == window.spike_status.loc[:, "left"].shift(-1)).iloc[:-1].all()

        elif direction == 'right':
            if window.spike_status.iloc[-1, get_sci('right')] < len(window.curve) - 1:
                return False

            return (window.spike_status.loc[:, "right"].shift(0) == window.spike_status.loc[:, "center"].shift(-1)).iloc[:-1].all()

        else:
            self.logger.error(role="Locator", message="direction must be 'left', 'right' or 'both'")
            raise RuntimeError("direction must be 'left', 'right' or 'both'")

    def _push_spike(
        self,
        window: Window,
        step: int = 1,
        direction: str = 'both',
    ) -> Window:
        """


        :param step:
        :param direction:

        :return:
        """
        while not self._is_terminate(window=window, direction=direction):
            self._step_spike(window=window, step=step, direction=direction)
            self._step_label(window=window)

        return window



class PenaltyComputer:

    @staticmethod
    def compute_penalty(
        window: Window,
        label_penalty: float = 10.0,
        spike_penalty: float = 0.1,
    ) -> float:
            """
            The penalty computation function

            :param window: The window describing the spike_status and label_status
            :param label_penalty: The weight of the missing-label-penalty
            :param spike_penalty: The weight of the expansion-width-penalty

            :return: The penalty term
            """

            spike_gap = np.mean(np.abs(window.spike_status.loc[:, 'right'].to_numpy() - window.spike_status.loc[:, 'left'].to_numpy()), dtype=np.float64)
            spike_pen = spike_gap * spike_penalty

            label_hit = np.mean(window.label_status.loc[:, 'hit'].to_numpy() == False, dtype=np.float64)
            label_pen = label_hit * label_penalty

            return spike_pen + label_pen

    @staticmethod
    def compute_penalties(
        window_configs: Tuple[Dict[str, Window], ...],
        label_penalty: float = 10.0,
        spike_penalty: float = 0.1,
    ) -> pd.DataFrame:

        penalty_table = pd.DataFrame(
            None,
            index=("cont", "cont_inv", "cont&inv", "both", "both_inv", "both&inv", "walk"),
            columns=("none", "right", "left", "both"),
            dtype=float
        )

        kernel = partial(PenaltyComputer.compute_penalty, label_penalty=label_penalty, spike_penalty=spike_penalty)

        for window_config in window_configs:
            for direction, window in window_config.items():
                if not window.valid:
                    continue

                if pd.isna(penalty_table.loc[window.beh, direction]):
                    penalty_table.loc[window.beh, direction] = 0.0

                penalty_table.loc[window.beh, direction] += kernel(window=window)

        penalty_table = penalty_table.dropna(how="all")
        penalty_table = penalty_table / len(list(filter(lambda x: x['none'].beh == window_configs[0]['none'].beh, window_configs)))

        return penalty_table



class Visualizer:

    @staticmethod
    def display_window(
        window: Window,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> None:
        """

        :param tgt_folder:
        :param window:

        :return:
        """

        main_signal = window.curve.loc[:, 'SUM']
        back_signal = window.curve.loc[:, list(set(window.curve.columns) - {"SUM"})]

        upper = main_signal.max()
        lower = main_signal.min()
        margin = abs(upper - lower) * 0.2

        plt.figure(figsize=(16, 9))

        plt.xlim(0, len(main_signal) - 1)
        pos = np.arange(0, len(main_signal), 1000).tolist()
        lab = main_signal.index[::1000].map(lambda x: f"{x:.2f}")
        plt.xticks(pos, lab)

        plt.plot(list(range(len(back_signal))), back_signal, lw=0.5, alpha=0.5, c='lightgray', label='others')
        plt.plot(list(range(len(main_signal))), main_signal, lw=1, alpha=1, c='dimgrey', label='signal')

        for spike in window.spike_status.to_dict("records"):
            plt.fill_between(
                [spike["left"], spike["right"]],
                [lower - margin, lower - margin],
                [upper + margin, upper + margin],
                alpha=0.2,
                color='crimson' if not spike['hit'] else 'lime'
            )
            plt.axvline(
                x=spike["center"],
                alpha=0.9,
                linestyle='--',
                label='peaks',
                color='crimson' if not spike['hit'] else 'darkgreen'
            )

        for lab in window.label_status.to_dict("records"):
            plt.axvline(x=lab['index'], color='turquoise', alpha=0.9, linestyle='--', label='labels')

        plt.grid(axis='y')
        plt.ylim(lower, upper)
        plt.title(f'{window.token} Signals & Labels')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        if tgt_folder:
            plt.savefig(Path(tgt_folder) / Path(f"penalty_window.pdf"))

        plt.show()
        plt.close()

    @staticmethod
    @check_folders("tgt_folder")
    def display_3d_penalty(
        table: pd.DataFrame,
        tgt_folder: None | str | os.PathLike | Path = None,
        dx=0.15,
        dy=0.15,
        left_till=0,
        rotation=0,
    ) -> None:
        """


        :param tgt_folder:
        :param table:
        :param dx:
        :param dy:
        :param left_till:
        :param rotation:

        :return: None
        """

        fig = plt.figure(figsize=(12, 9))
        ax = cast(Axes3D, fig.add_subplot(projection='3d'))

        xs = np.arange(table.shape[1])
        ys = np.arange(table.shape[0])
        x_pos, y_pos = np.meshgrid(xs, ys, indexing='xy')
        x_pos, y_pos = x_pos.ravel(), y_pos.ravel()
        z_pos = np.zeros_like(x_pos)

        dxs = np.full_like(z_pos, dx, dtype=float)
        dys = np.full_like(z_pos, dy, dtype=float)

        # the invisible left ax ====================
        ax.set_xlabel('Causality')
        ax.set_ylabel('Behavior')
        ax.set_zlabel('Penalty')
        ax.set_title("Causality-Penalty Comparison")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        ax.set_xticks(xs)
        ax.set_xticklabels(table.columns)
        ax.set_yticks(ys)
        ax.set_yticklabels(table.index)

        ax.view_init(elev=15, azim=-55 + rotation)
        ax.set_proj_type('ortho')

        _df = table.copy()
        _df.iloc[:, list(range(left_till + 1, table.shape[1]))] = 0
        _dz = _df.values.ravel()

        ax.bar3d(
            x_pos - dx / 2,
            y_pos - dy / 2,
            z_pos,
            dxs,
            dys,
            _dz,
            shade=True,
            color=["#d1d1d1" if h > 0 else (0, 0, 0, 0) for h in _dz]
        )

        z_ticks = ax.get_zticks()
        z_min, z_max = ax.get_zlim()
        ax.set_zticks([])

        # the right ax ==========================
        df_ = table.copy()
        df_.iloc[:, list(range(0, left_till + 1))] = 0
        dz_ = df_.values.ravel()

        ax_ = cast(
            Axes3D,
            fig.add_axes(
                ax.get_position().bounds,
                projection='3d',
                label='left_inv'
            )
        )

        ax_.grid(False)
        ax_.patch.set_alpha(0)
        ax_.xaxis.set_visible(False)
        ax_.yaxis.set_visible(False)
        ax_.zaxis.set_visible(False)
        for axis in (ax_.xaxis, ax_.yaxis, ax_.zaxis):
            axis.set_pane_color((1, 1, 1, 0))

        ax_.bar3d(
            x_pos - dx / 2,
            y_pos - dy / 2,
            z_pos,
            dxs,
            dys,
            dz_,
            shade=True,
            color=["#56a4d9" if h > 0 else (0, 0, 0, 0) for h in dz_]
        )

        ax_.set_xticks(xs)
        ax_.set_xticklabels(table.columns)
        ax_.set_yticks(ys)
        ax_.set_yticklabels(table.index)

        ax_.view_init(elev=15, azim=-55 + rotation)
        ax_.set_proj_type('ortho')

        # the left ax ==============================
        _ax = cast(
            Axes3D,
            fig.add_axes(
                ax.get_position().bounds,
                projection='3d',
                label='left_clr'
            )
        )
        _ax.grid(False)
        _ax.patch.set_alpha(0)
        _ax.xaxis.set_visible(False)
        _ax.yaxis.set_visible(False)
        for axis in (_ax.xaxis, _ax.yaxis, _ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))

        _ax.set_xlabel('')
        _ax.set_ylabel('')
        _ax.set_zlabel('Penalty')
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_zticks(z_ticks)
        _ax.set_zlim(z_min, z_max)
        _ax.xaxis.set_visible(False) # changed
        _ax.yaxis.set_visible(False) # changed

        _ax.view_init(elev=15, azim=270 - 55 + rotation)
        _ax.set_proj_type('ortho')

        if tgt_folder:
            plt.savefig(Path(tgt_folder) / Path(f"penalty_3d.pdf"))

        plt.show()
        plt.close()



