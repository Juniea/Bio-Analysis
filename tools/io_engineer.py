import random
from typing import Callable, Any
import os
from pathlib import Path
import re


from tools.logger import SyncLogger


def check_folders(*path_args: str) -> Callable:
    """
    The decorator generator that makes sure all dependent folders exist before the decorated function is called

    :param path_args: All dependent folders

    :return: The decorated function
    """

    logger = SyncLogger(level="INFO")

    def decorator(funct: Callable) -> Callable:

        def wrapper(*args, **kwargs) -> Any:
            # check and create the dependencies if needed
            for path_arg in path_args:
                path = kwargs.get(path_arg)

                if path is None:
                    continue

                try:
                    path = Path(path)
                except Exception as e:
                    logger.warning(role=f"f{funct.__name__}", message=f"no such kwarg at function calling: {path_arg}")
                else:
                    if path is None:
                        logger.warning(role=f"f{funct.__name__}", message=f"no such kwarg at function calling: {path_arg}")
                    elif not path.exists():
                        logger.info(role=f"f{funct.__name__}", message=f"Automatic created {path}")
                        path.mkdir(parents=True, exist_ok=True)

            # core function execution
            return funct(*args, **kwargs)

        return wrapper

    return decorator


def fetch_tokens(
    folder: str | os.PathLike | Path,
    shard: str = "all"
) -> tuple[str, ...]:
    """
    Fetch all valid tokens from a given folder (non-recursive)

    :param folder: The path of the folder
    :param shard: Which shard the folder belongs to

    :return: The sorted tuple of the tokens
    """
    logger = SyncLogger(level="INFO")

    pattern = re.compile(r"^(K\d{3}_[AB]_[A-Za-z]\d)_[A-Za-z]$")

    matched_tokens = []

    for file in folder.iterdir():
        if file.is_file():
            stem = file.stem
            m = pattern.match(stem)
            if m:
                token = m.group(1)
                if token not in matched_tokens:
                    matched_tokens.append(token)

    good_tokens = [
        "K215_B_D2",
        "K215_B_D4",
        "K215_B_D8",

        "K223_A_D1",
        "K223_A_D4",
        "K223_A_D7",

        "K232_A_D1",  # Sparce cont&inv exclude at fitting cont&inv
        "K232_A_D4",
        "K232_A_D7",
        "K232_B_D2",
        "K232_B_D5",
        "K232_B_D8",

        "K228_B_D2",  # Sparce cont_inv exclude at fitting cont_inv
        "K228_B_D5",
        "K228_B_D8"
    ]
    good_tokens = set(good_tokens)

    if shard == "all":
        logger.info(role="fetch_tokens", message="fetching all tokens")
        return tuple(sorted(matched_tokens))
    elif shard == "good":
        logger.info(role="fetch_tokens", message="fetching good tokens only")
        return tuple(sorted(filter(lambda x: x in good_tokens, matched_tokens)))
    elif shard == "other":
        logger.info(role="fetch_tokens", message="fetching other tokens only")
        return tuple(sorted(filter(lambda x: x not in good_tokens, matched_tokens)))
    elif shard.startswith("test"):
        if shard[4:].isnumeric():
            logger.info(role="fetch_tokens", message="fetching test tokens only")
            return tuple(sorted(random.sample(matched_tokens, k=int(shard[4:]))))
        else:
            logger.error(role="fetch_tokens", message="invalid shard name when fetching tokens")
            raise RuntimeError(f"invalid shard name when fetching tokens")
    else:
        logger.error(role="fetch_tokens", message="invalid shard name when fetching tokens")
        raise RuntimeError(f"invalid shard name when fetching tokens")
