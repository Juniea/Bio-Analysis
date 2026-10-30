import os
import time
from pathlib import Path
from dotenv import load_dotenv

from copy import deepcopy
import random
import pandas as pd
import numpy as np

from tools.logger import SyncLogger
from tools.io_engineer import fetch_tokens
from src_predict.preprocessor import Preprocessor

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))  # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))  # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))  # path of the memory data directory


# global test report logger
logger = SyncLogger(level="DEBUG")


def test_preprocessor_determinism():

    def generate_random_config(token, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        tvt_ratio = np.random.rand(3)
        tvt_ratio = tvt_ratio / sum(tvt_ratio)

        return {
            "beh_path": PATH_DATA / Path(f'{token}_B.csv'),
            "sig_path": PATH_DATA / Path(f'{token}_S.csv'),
            "tvt_ratio": tuple(tvt_ratio.tolist()),
            "shuffle": random.choice([True, False]),
            "normalizer": random.choice(['z-score', 'min-max', None]),
            "agg": random.choice([True, False]),
            "smote": random.choice([random.random(), None]),
            "resize": random.choice([True, False]),
            "target": random.choice(['cont', 'cont&inv', 'cont_inv', 'both', 'both_inv', 'both&inv']),
            "log_level": "SILENCE"
        }

    tokens = fetch_tokens(folder=PATH_DATA, shard="all")
    count = 0
    for token in tokens:
        config = generate_random_config(token, int(time.time()))
        try:
            prep1 = Preprocessor(**config)
            prep2 = Preprocessor(**config)
        except Exception as e:
            continue

        try:
            pd.testing.assert_frame_equal(prep1.get_origin_total(shard="all"), prep2.get_origin_total(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_after_total(shard="all"), prep2.get_after_total(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_train(shard="all"), prep2.get_train(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_valid(shard="all"), prep2.get_valid(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_test(shard="all"), prep2.get_test(shard="all"))
            count += 1
        except AssertionError as e:
            logger.debug(role="test_preprocessor_determinism", message=f"Failed: {e}")
            return

    logger.debug(role="test_preprocessor_determinism", message=f"Success [{count}/{len(tokens)}] tested")


if __name__ == '__main__':
    test_preprocessor_determinism()


