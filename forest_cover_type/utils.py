﻿import sys, configparser
from ast import literal_eval
from distutils.util import strtobool
from pathlib import Path
from loguru import logger  # type:ignore
import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes.
    Allows to get an attr by settings.dataset_path instead of settings["dataset_path"]
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def process_settings(settings):
    # set some needed default values if not exist
    settings.coef_1 = 1.4
    settings.coef_2 = 3.7
    settings.coef_3 = 3.8
    settings.mode = settings.get("mode", "default_mode")
    settings.n_splits = settings.get("n_splits", 3)
    settings.feature_engineering = settings.get("feature_engineering", None)
    settings.n_jobs = settings.get("n_jobs", -1)
    settings.clf_n_estimators = settings.get("clf_n_estimators", 200)
    settings.use_booster = settings.get("use_booster", False)
    settings.booster_n_estimators_1 = settings.get("booster_n_estimators_1", 200)
    settings.booster_n_estimators_2 = settings.get("booster_n_estimators_2", 200)
    settings.random_state = settings.get("SEED", 0)
    settings.model_path = Path(settings.get("model_path", "data/default_model_path"))
    settings.save_if_not_exists = settings.get("save_if_not_exists", False)
    settings.load_if_exists = settings.get("load_if_exists", False)
    settings.load_test_csv = settings.get("load_test_csv", True)
    # load from ini if provided
    if settings.train_cfg is not None:
        config = configparser.ConfigParser()
        config.read(settings.train_cfg, encoding="utf-8")
        settings.mode = config.get("Common", "mode", fallback=settings.mode)
        settings.n_splits = int(config.get("Common", "n_splits", fallback=settings.n_splits))
        settings.feature_engineering = config.get(
            "Common", "feature_engineering", fallback=settings.feature_engineering
        )
        settings.model_path = Path(config.get("Common", "model_path", fallback=settings.model_path))
        settings.save_if_not_exists = strtobool(
            config.get("Common", "save_if_not_exists", fallback=str(settings.save_if_not_exists))
        )
        settings.load_if_exists = strtobool(
            config.get("Common", "load_if_exists", fallback=str(settings.load_if_exists))
        )
        settings.n_jobs = int(config.get("Common", "n_jobs", fallback=settings.n_jobs))
        settings.random_state = int(config.get("Common", "random_state", fallback=settings.random_state))
        settings.clf_n_estimators = int(
            config.get("ExtraTreesClassifier", "n_estimators", fallback=settings.clf_n_estimators)
        )
        settings.use_booster = strtobool(config.get("Booster", "enable", fallback=str(settings.use_booster)))
        settings.booster_n_estimators_1 = int(
            config.get("Booster", "n_estimators_1", fallback=settings.booster_n_estimators_1)
        )
        settings.booster_n_estimators_2 = int(
            config.get("Booster", "n_estimators_2", fallback=settings.booster_n_estimators_2)
        )

        settings["runs"] = dotdict({})
        for section in config.sections():
            if section.startswith("run_"):
                settings["runs"][section] = dotdict({})
                for key in config[section]:
                    # print("config[section][key]:", config[section][key])
                    try:
                        # try to parse array/dict/etc
                        settings.runs[section][key] = literal_eval(config[section][key])
                    except ValueError:
                        # looks like it's a string
                        settings.runs[section][key] = config[section][key]

        if settings.mode == "kfold" and len(settings.runs.keys()) == 0:
            logger.error("kfold config must have [run_N] section")
            assert len(settings.runs.keys()) > 0

    if settings.use_logfile:
        logger_config = {
            "handlers": [
                {"sink": sys.stdout, "format": "<green>{module}</green> {message}"},
                {"sink": "file.log", "format": "{time:YYYY-MM-DD HH:mm:ss} {module} {message}"},
            ]
        }
    else:
        logger_config = {"handlers": [{"sink": sys.stdout, "format": "<green>{module}</green> {message}"}]}
    logger.configure(**logger_config)


def autoreload():
    """This enables auto-reload mode for qtconsole to reload module after the source code was changed.
    Example: %run -m forest_cover_type -a
    """
    get_ipython().run_line_magic("load_ext", "autoreload")  # type:ignore
    get_ipython().run_line_magic("autoreload", "2")  # type:ignore
    # also set these options for pretty output
    np.set_printoptions(
        precision=3,
        suppress=True,
        linewidth=115,
        threshold=1000,
        formatter=dict(float_kind=lambda x: "%6.3f" % x),
    )
