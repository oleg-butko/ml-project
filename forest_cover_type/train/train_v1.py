import sys, configparser
from distutils.util import strtobool
from pathlib import Path
from joblib import dump, load
from loguru import logger  # type:ignore
from sklearn import ensemble


def run(settings, dataframes):
    assert len(dataframes) == 3
    # defaults, can be overwritten by settings.train_cfg (train.ini)
    n_jobs = -1
    clf_n_estimators = settings.clf_n_estimators if settings.clf_n_estimators else 100
    use_booster = False
    booster_n_estimators_1 = 200
    booster_n_estimators_2 = 200
    random_state = settings.SEED
    model_path = Path("data/default_model_path")
    save_if_not_exists = False
    load_if_exists = False

    if settings.train_cfg is not None:
        config = configparser.ConfigParser()
        config.read(settings.train_cfg, encoding="utf-8")
        model_path = Path(config.get("Common", "model_path", fallback=model_path))
        save_if_not_exists = strtobool(
            config.get("Common", "save_if_not_exists", fallback=str(save_if_not_exists))
        )
        load_if_exists = strtobool(config.get("Common", "load_if_exists", fallback=str(load_if_exists)))
        n_jobs = int(config.get("Common", "n_jobs", fallback=n_jobs))
        random_state = int(config.get("Common", "random_state", fallback=random_state))
        clf_n_estimators = int(config.get("ExtraTreesClassifier", "n_estimators", fallback=clf_n_estimators))
        use_booster = strtobool(config.get("Booster", "enable", fallback=str(use_booster)))
        # send this option to the predict step
        settings.use_booster = use_booster
        booster_n_estimators_1 = int(config.get("Booster", "n_estimators_1", fallback=booster_n_estimators_1))
        booster_n_estimators_2 = int(config.get("Booster", "n_estimators_2", fallback=booster_n_estimators_2))

    clf = ensemble.ExtraTreesClassifier(
        n_estimators=clf_n_estimators, max_depth=6, n_jobs=n_jobs, random_state=random_state
    )
    if not use_booster:
        if load_if_exists and model_path.is_file() and model_path.exists():
            clf = load(model_path)
            logger.info(f"Loaded model from path: {model_path}")

    X_train, y = dataframes[0]
    logger.info(f"clf.fit(X_train, y), X_train.shape: {X_train.shape}")
    clf.fit(X_train, y)

    if use_booster:
        clf_1_2 = ensemble.RandomForestClassifier(
            n_estimators=booster_n_estimators_1, n_jobs=n_jobs, random_state=random_state
        )
        X_train_1_2, y_1_2 = dataframes[1]
        logger.info(f"clf_1_2.fit(X_train_1_2, y_1_2), X_train_1_2.shape: {X_train_1_2.shape}")
        clf_1_2.fit(X_train_1_2, y_1_2)
        clf_3_4_6 = ensemble.RandomForestClassifier(
            n_estimators=booster_n_estimators_2, n_jobs=n_jobs, random_state=random_state
        )
        X_train_3_4_6, y_3_4_6 = dataframes[2]
        logger.info(f"clf_3_4_6.fit(X_train_3_4_6, y_1_2), X_train_3_4_6.shape: {X_train_3_4_6.shape}")
        clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
        return {"clf": clf, "clf_1_2": clf_1_2, "clf_3_4_6": clf_3_4_6}
    else:
        if save_if_not_exists:
            if model_path.exists():
                pass
            else:
                dump(clf, model_path)
                if model_path.is_file() and model_path.exists():
                    logger.info(f"Model was saved to: {model_path}")
                else:
                    logger.error(f"Error while saving model with path: {model_path}")
        return {"clf": clf}
