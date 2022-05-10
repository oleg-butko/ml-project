import sys, configparser
from ast import literal_eval
from distutils.util import strtobool
from pathlib import Path
from loguru import logger  # type:ignore


class dotdict(dict):
    """dot.notation access to dictionary attributes.
    Allows to get an attr by settings.dataset_path instead of settings["dataset_path"]
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def process_settings(settings):
    # set some needed default values if not exists
    settings.mode = settings.get("mode", "default_mode")
    settings.n_splits = settings.get("n_splits", 3)
    settings.n_jobs = settings.get("n_jobs", -1)
    settings.clf_n_estimators = settings.get("clf_n_estimators", 4)
    settings.use_booster = settings.get("use_booster", False)
    settings.booster_n_estimators_1 = settings.get("booster_n_estimators_1", 200)
    settings.booster_n_estimators_2 = settings.get("booster_n_estimators_2", 200)
    settings.random_state = settings.get("SEED", 0)
    settings.model_path = Path(settings.get("model_path", "data/default_model_path"))
    settings.save_if_not_exists = settings.get("save_if_not_exists", False)
    settings.load_if_exists = settings.get("load_if_exists", False)
    # load from ini if provided
    if settings.train_cfg is not None:
        config = configparser.ConfigParser()
        config.read(settings.train_cfg, encoding="utf-8")
        settings.mode = config.get("Common", "mode", fallback=settings.mode)
        settings.n_splits = int(config.get("Common", "n_splits", fallback=settings.n_splits))
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
                    settings.runs[section][key] = literal_eval(config[section][key])

        if settings.mode == "kfold" and len(settings.runs.keys()) == 0:
            logger.error(f"kfold config must have [run_N] section")
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
