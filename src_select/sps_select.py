import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Callable, Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools import ClassificationFeatureEngineer, SyncLogger, check_folders, fetch_tokens
from .preprocessor import Preprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


class SPSSelector:
    @check_folders("sig_folder", "beh_folder")
    def __init__(
        self,
        sig_folder: str | os.PathLike | Path,
        beh_folder: str | os.PathLike | Path,
        seed: int = 42,
        log_level: str = "INFO",
    ) -> None:
        self.sig_folder = Path(sig_folder)
        self.beh_folder = Path(beh_folder)
        self.seed = seed

        self.logger = SyncLogger(level=log_level)

    @staticmethod
    def find_target(array, funct=lambda x: x == 1):
        array = array.to_list()
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
                        windows.append((left, right))
                    else:
                        windows.append((left, right - 1))

                left = right

        return windows

    @staticmethod
    def diff_dict(dict_ref: Dict[str, Any], dict_new: Dict[str, Any]):
        diff_collector = {}

        for key_ref, value_ref in dict_ref.items():
            if key_ref not in dict_new:
                diff_collector[key_ref] = (value_ref, None)

            elif value_ref != dict_new[key_ref]:
                diff_collector[key_ref] = (value_ref, dict_new[key_ref])

        return diff_collector

    @check_folders("tgt_folder")
    def event_heatmap(
        self,
        token: str,
        beh: str                                        = "cont",
        cells: None | Tuple[int, ...] | Tuple[str, ...] = None,
        outer_width: int                                = 8,
        inner_width: Tuple[int, int]                    = (-50, 50),
        cmap_sel: str                                   = "afmhot",
        cmap_oth: str                                   = "viridis",
        center_line: bool                               = True,
        viz: bool                                       = True,
        plt_folder: None | str | os.PathLike | Path     = None,
    ) -> None:
        """
        The method generating all event heatmaps for a given token

        :param token: The token of the subject
        :param beh: The behavior of the subject
        :param cells: The cells that are selected as associated
        :param outer_width: The column number of the event heatmap matrix
        :param inner_width: The width of the observation window
        :param cmap_sel: The color map of the selected cells' event heatmap
        :param cmap_oth: The color map of the non-selected cells' event heatmap
        :param center_line: The toggle of the center dashed line
        :param viz: The toggle of the visualization
        :param plt_folder: The target folder where the event heatmap will be saved

        :return: None
        """
        prep = Preprocessor(
            beh_path=self.beh_folder / Path(f"{token}_B.csv"),
            sig_path=self.sig_folder / Path(f"{token}_S.csv"),
            target=beh,
            tvt_ratio=(1.0, 0.0, 0.0),
            shuffle=False,
            normalizer='z-score',
            agg=False,
            smote=None,
            resize=True,
            log_level="WARNING"
        )
        x_tot = prep.get_origin_total(shard='x')
        y_tot = prep.get_origin_total(shard='y')
        y_inv = SPSSelector.find_target(y_tot)
        starts = list(map(lambda x: x[0], y_inv))

        # parse "X000N" cells into corresponding column indexes
        if cells is not None and cells:
            if isinstance(cells[0], str):
                cells = list(map(lambda x: x_tot.columns.get_loc(x), cells))
        else:
            cells = []

        # init the selected cells' and non-selected cells' event heatmap records
        sel_cells_lists = {c: [] for c in cells}
        oth_cells_lists = {o: [] for o in list(set(list(range(0, x_tot.shape[1]))) - set(cells))}
        sel_cells_matrices = {}
        oth_cells_matrices = {}

        # populate selected cells' and non-selected cells' event heatmap records
        for ele in range(0, x_tot.shape[1]):
            for start in starts:
                fst = max(start + inner_width[0], 0)
                snd = min(start + inner_width[1], len(x_tot) - 1)

                # skip observation widows whose left boarder or right boarder exceeds the endpoints of the signal data
                if abs(snd - fst) == abs(inner_width[1] - inner_width[0]):
                    temp = x_tot.iloc[fst: snd + 1, ele].to_list()

                    if ele in sel_cells_lists.keys():
                        sel_cells_lists[ele].append(temp)
                    else:
                        oth_cells_lists[ele].append(temp)

            if ele in sel_cells_lists.keys():
                sel_cells_matrices[ele] = np.stack(sel_cells_lists[ele], axis=0)
            else:
                oth_cells_matrices[ele] = np.stack(oth_cells_lists[ele], axis=0)

        del sel_cells_lists
        del oth_cells_lists

        # min-max normalize so that the signals are all compressed into 0-1 range
        for key, matrix in sel_cells_matrices.items():
            matrix_min = matrix.min()
            matrix_max = matrix.max()

            sel_cells_matrices[key] = (matrix - matrix_min) / (matrix_max - matrix_min)

        for key, matrix in oth_cells_matrices.items():
            matrix_min = matrix.min()
            matrix_max = matrix.max()

            oth_cells_matrices[key] = (matrix - matrix_min) / (matrix_max - matrix_min)


        # init the plot canvas configuration
        outer_height = (
                int(np.ceil(len(sel_cells_matrices) / outer_width)) +
                int(np.ceil(len(oth_cells_matrices) / outer_width))
        )
        plt.figure(figsize=(3 + outer_width * 3, outer_height * 3))
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.92,
            bottom=0.05,
            wspace=0.3,
            hspace=0.5
        )

        sel_cells_matrices = sorted([(key, value) for key, value in sel_cells_matrices.items()], key=lambda t: t[0])
        oth_cells_matrices = sorted([(key, value) for key, value in oth_cells_matrices.items()], key=lambda t: t[0])
        counter = 0
        for is_other, (cur_cells_matrices, cmap_cur) in enumerate(zip(
                [sel_cells_matrices, oth_cells_matrices],
                [cmap_sel, cmap_oth]
        )):
            for idx, (key, value) in enumerate(cur_cells_matrices):
                counter += 1

                plt.subplot(outer_height, outer_width, counter)

                im = plt.imshow(
                    value,
                    aspect='auto',
                    cmap=cmap_cur,
                    origin='upper',
                    extent=(inner_width[0], inner_width[1], value.shape[0], 0)
                )
                plt.colorbar(im)

                if center_line:
                    plt.axvline(x=abs(inner_width[0] + inner_width[1]) / 2, color='white', linewidth=1)

                plt.xlim(inner_width[0], inner_width[1])
                plt.xticks(
                    range(inner_width[0], inner_width[1] + 1, 25),
                    labels=list(map(lambda x: str(x), range(inner_width[0], inner_width[1] + 1, 25)))
                )
                plt.tick_params(axis='y', labelleft=False)

                plt.title(f"{x_tot.columns[key]}")
                plt.xlabel("0.1 seconds")
                plt.ylabel("event window")

            if counter % outer_width != 0:
                counter = int(np.ceil(counter / outer_width)) * outer_width

        plt.suptitle(f"{token}-{beh} - selected/unselected event heatmap")

        plt.savefig(plt_folder / Path(f"{token}_{beh}_event-heatmap.pdf"))
        if viz:
            plt.show()
        plt.close()

        self.logger.info(role="SPSSelector", message=f"event heatmap for {token} on {beh} completed")

    @check_folders("tgt_folder")
    def event_curve(
        self,
        token: str,
        beh: str                                        = "cont",
        cells: None | Tuple[int, ...] | Tuple[str, ...] = None,
        outer_width: int                                = 8,
        inner_width: Tuple[int, int]                    = (-50, 50),
        cmap_sel: str                                   = "crimson",
        cmap_oth: str                                   = "royalblue",
        center_line: bool                               = True,
        viz: bool                                       = True,
        plt_folder: None | str | os.PathLike | Path     = None,
    ) -> None:
        """
        The method generating all event heatmaps for a given token

        :param token: The token of the subject
        :param beh: The behavior of the subject
        :param cells: The cells that are selected as associated
        :param outer_width: The column number of the event heatmap matrix
        :param inner_width: The width of the observation window
        :param cmap_sel: The color map of the selected cells' event heatmap
        :param cmap_oth: The color map of the non-selected cells' event heatmap
        :param center_line: The toggle of the center dashed line
        :param viz: The toggle of the visualization
        :param plt_folder: The target folder where the event heatmap will be saved

        :return: None
        """
        prep = Preprocessor(
            beh_path=self.beh_folder / Path(f"{token}_B.csv"),
            sig_path=self.sig_folder / Path(f"{token}_S.csv"),
            target=beh,
            tvt_ratio=(1.0, 0.0, 0.0),
            shuffle=False,
            normalizer='z-score',
            agg=False,
            smote=None,
            resize=True,
            log_level="WARNING"
        )
        x_tot = prep.get_origin_total(shard='x')
        y_tot = prep.get_origin_total(shard='y')
        y_inv = SPSSelector.find_target(y_tot)
        starts = list(map(lambda x: x[0], y_inv))

        # parse "X000N" cells into corresponding column indexes
        if cells is not None and cells:
            if isinstance(cells[0], str):
                cells = list(map(lambda x: x_tot.columns.get_loc(x), cells))
        else:
            cells = []

        # init the selected cells' and non-selected cells' event heatmap records
        sel_cells_curve = {c: np.zeros(abs(inner_width[1] - inner_width[0]) + 1) for c in cells}
        oth_cells_curve = {o: np.zeros(abs(inner_width[1] - inner_width[0]) + 1) for o in list(set(list(range(0, x_tot.shape[1]))) - set(cells))}

        # populate selected cells' and non-selected cells' event heatmap records
        for ele in range(0, x_tot.shape[1]):
            for start in starts:
                fst = max(start + inner_width[0], 0)
                snd = min(start + inner_width[1], len(x_tot) - 1)

                # skip observation widows whose left boarder or right boarder exceeds the endpoints of the signal data
                if abs(snd - fst) == abs(inner_width[1] - inner_width[0]):
                    temp = x_tot.iloc[fst: snd + 1, ele].to_numpy()

                    if ele in sel_cells_curve.keys():
                        sel_cells_curve[ele] += temp
                    else:
                        oth_cells_curve[ele] += temp


        # min-max normalize so that the signals are all compressed into 0-1 range
        for key, curve in sel_cells_curve.items():
            curve_min = curve.min()
            curve_max = curve.max()

            sel_cells_curve[key] = (curve - curve_min) / (curve_max - curve_min)

        for key, curve in oth_cells_curve.items():
            curve_min = curve.min()
            curve_max = curve.max()

            oth_cells_curve[key] = (curve - curve_min) / (curve_max - curve_min)

        # init the plot canvas configuration
        outer_height = (
                int(np.ceil(len(sel_cells_curve) / outer_width)) +
                int(np.ceil(len(oth_cells_curve) / outer_width))
        )
        plt.figure(figsize=(3 + outer_width * 3, outer_height * 3))
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.92,
            bottom=0.05,
            wspace=0.3,
            hspace=0.5
        )

        sel_cells_matrices = sorted([(key, value) for key, value in sel_cells_curve.items()], key=lambda t: t[0])
        oth_cells_matrices = sorted([(key, value) for key, value in oth_cells_curve.items()], key=lambda t: t[0])
        counter = 0
        for is_other, (cur_cells_matrices, cmap_cur) in enumerate(zip(
                [sel_cells_matrices, oth_cells_matrices],
                [cmap_sel, cmap_oth]
        )):
            for idx, (key, value) in enumerate(cur_cells_matrices):
                counter += 1

                plt.subplot(outer_height, outer_width, counter)

                plt.plot(list(range(inner_width[0], inner_width[1] + 1)), value, color=cmap_cur)
                plt.fill_between(
                    x=list(range(inner_width[0], inner_width[1] + 1)),
                    y1=value,
                    y2=0,
                    color=cmap_cur,
                    alpha=0.3,
                )

                if center_line:
                    plt.axvline(x=abs(inner_width[0] + inner_width[1]) / 2, color='black', linewidth=1)

                plt.xlim(inner_width[0], inner_width[1])
                plt.xticks(
                    range(inner_width[0], inner_width[1] + 1, 25),
                    labels=list(map(lambda x: str(x), range(inner_width[0], inner_width[1] + 1, 25)))
                )
                plt.ylim(0.0, 1.0)
                plt.yticks(
                    [0, 1],
                    labels=['0', '1']
                )

                plt.title(f"{x_tot.columns[key]}")
                plt.xlabel("0.1 seconds")
                plt.ylabel("signal significance")

            if counter % outer_width != 0:
                counter = int(np.ceil(counter / outer_width)) * outer_width

        plt.suptitle(f"{token}-{beh} - selected/unselected event curve")

        plt.savefig(plt_folder / Path(f"{token}_{beh}_event-curves.pdf"))
        if viz:
            plt.show()
        plt.close()

        self.logger.info(role="SPSSelector", message=f"event curve for {token} on {beh} completed")

    @check_folders("tgt_folder", "plt_folder")
    def sps_select(
        self,
        token: str,
        beh: str                                    = "cont",
        standard: int                               = 4,
        threshold: float                            = 0.6,
        score: str                                  = 'f1',
        resize: bool                                = True,
        flip: bool                                  = False,
        rerun: bool                                 = False,
        viz: bool                                   = True,
        tgt_folder: None | str | os.PathLike | Path = None,
        plt_folder: None | str | os.PathLike | Path = None,
    ) -> Tuple[int, ...]:
        """
        The method that selects the most associated cells that by itself can predict the most of the behaviors

        :param token: The token of the subject
        :param beh: The behavior of the subject
        :param standard: The minimum votes a cell has to get for being selected
        :param threshold: The minimum score a model has to get for voting
        :param score: The score type for prediction
        :param resize: The toggle for shrinking / expanding the signals
        :param flip: The toggle for flipping invalid scores to valid ones
        :param rerun: The toggle for re-running the cell selection
        :param viz: The toggle for visualising the voting results
        :param tgt_folder: The folder sorting the cached scoring results
        :param plt_folder: The folder sorting the visualization of the voting results

        :return: The indexes of the selected cells
        """

        models = ['LR', 'SVM', 'RF', 'XGB', 'LGB']

        # fetch the scoring matrix if exists
        try:
            if rerun:
                raise Exception("rerun hack")

            with open(Path(tgt_folder) / Path(f"{token}_C.json"), 'r') as fp:
                lookup_ref = json.load(fp)

            if (
                score.lower().strip() != lookup_ref.get("score", None).lower().strip() or
                resize != lookup_ref.get("resize", None)
            ):
                raise Exception("rerun hack")

            score_mat = pd.read_csv(Path(tgt_folder) / Path(f"{token}_V.csv"))

        # compute the scoring matrix if not exists
        except Exception as e:
            prep = Preprocessor(
                beh_path=self.beh_folder / Path(f"{token}_B.csv"),
                sig_path=self.sig_folder / Path(f"{token}_S.csv"),
                target=beh,
                tvt_ratio=(1.0, 0.0, 0.0),
                shuffle=True,
                normalizer='z-score',
                agg=False,
                smote='auto',
                resize=resize,
                log_level="WARNING"
            )
            x_tot = prep.get_after_total(shard='x')
            y_tot = prep.get_after_total(shard='y')

            eng = ClassificationFeatureEngineer(x_tot, y_tot, self.seed)
            score_mat = eng.single_score_select(
                models=models,
                cv=10,
                threshold=0.595,
                score=score,
                normalize='z-score',
                parallel=True,
                verbose=False,
            )

            score_mat.to_csv(Path(tgt_folder) / Path(f"{token}_V.csv"), index=False)
            score_mat = score_mat.reset_index(drop=True)

        # flip the invalid scoring
        if flip:
            score_mat = score_mat.map(lambda x: 1.0 - x if x < 0.5 else x)

        # plot the sps scoring bar plot
        n = len(models)
        cmap_reds = plt.get_cmap('Reds')
        colors_reds = [cmap_reds(i / (n + 2)) for i in range(2, n + 2)]

        ax = score_mat.plot(
            kind='bar',
            title=f"{token}-{beh} {score.title()} Scoring Bars",
            xlabel='Cells',
            ylabel='Score',
            figsize=(16, 9),
            color=colors_reds,
        )
        ax.axhline(
            y=threshold,
            color='black',
            linestyle='--',
            linewidth=1
        )

        if viz:
            plt.show()

        if plt_folder:
            plt.savefig(Path(plt_folder) / Path(f"{token}-{beh}_sps-scoring.pdf"))
        plt.close()

        # get the selected cells that passes the filtering
        score_mask = score_mat >= threshold
        hits = score_mask.sum(axis=1).astype(int)
        selected = score_mask.loc[hits >= standard].index.to_list()

        # save the SPS-Selection configuration into cache
        if tgt_folder:
            path_save = Path(tgt_folder) / Path(f"{token}_C.json")
            lookup = {
                "models": models,
                "threshold": threshold,
                "standard": standard,
                "score": score,
                "flip": flip,
                "selected_index": selected,
                "resize": resize
            }

            try:
                with open(path_save, 'r') as fp:
                    lookup_ref = json.load(fp)

            except Exception as e:
                self.logger.info(role="SPSSelector", message=f"SPS-Selection on {token} for {beh} inited.")

            else:
                config_diff = " | ".join([
                    f"{key}: {value[0]} -> {value[1] if value[1] is not None else '_'}" for key, value in SPSSelector.diff_dict(lookup_ref, lookup).items()
                ])
                self.logger.info(role="SPSSelector", message=f"SPS-Selection on {token} for {beh} updated: {config_diff if config_diff else '_'}")

            with open(path_save, 'w') as fp:
                json.dump(lookup, fp, indent=4)

        return selected