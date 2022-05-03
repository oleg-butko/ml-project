import sys, configparser, distutils
from pathlib import Path
from joblib import dump, load
import click
from sklearn import ensemble


def run(settings, dataframes):
    assert len(dataframes) == 3
    # defaults, can be overwritten by train.ini
    n_jobs = -1
    clf_n_estimators = 100
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
        save_if_not_exists = distutils.util.strtobool(
            config.get("Common", "save_if_not_exists", fallback=save_if_not_exists)
        )
        load_if_exists = distutils.util.strtobool(
            config.get("Common", "load_if_exists", fallback=load_if_exists)
        )
        n_jobs = int(config.get("Common", "n_jobs", fallback=n_jobs))
        random_state = int(config.get("Common", "random_state", fallback=random_state))
        clf_n_estimators = int(config.get("ExtraTreesClassifier", "n_estimators", fallback=clf_n_estimators))
        use_booster = distutils.util.strtobool(config.get("Booster", "enable", fallback=use_booster))
        # send this option to the predict step
        settings.use_booster = use_booster
        booster_n_estimators_1 = int(config.get("Booster", "n_estimators_1", fallback=booster_n_estimators_1))
        booster_n_estimators_2 = int(config.get("Booster", "n_estimators_2", fallback=booster_n_estimators_2))

    clf = ensemble.ExtraTreesClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    X_train, y = dataframes[0]

    if not use_booster:
        if load_if_exists and model_path.is_file() and model_path.exists():
            clf = load(model_path)
            print("Loaded model from path:", model_path)

    clf.fit(X_train, y)

    if use_booster:
        clf_1_2 = ensemble.RandomForestClassifier(
            n_estimators=booster_n_estimators_1, n_jobs=n_jobs, random_state=random_state
        )
        X_train_1_2, y_1_2 = dataframes[1]
        clf_1_2.fit(X_train_1_2, y_1_2)
        clf_3_4_6 = ensemble.RandomForestClassifier(
            n_estimators=booster_n_estimators_2, n_jobs=n_jobs, random_state=random_state
        )
        X_train_3_4_6, y_3_4_6 = dataframes[2]
        clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
        return {"clf": clf, "clf_1_2": clf_1_2, "clf_3_4_6": clf_3_4_6}
    else:
        if save_if_not_exists:
            if model_path.exists():
                pass
            else:
                dump(clf, model_path)
                if model_path.is_file() and model_path.exists():
                    click.echo(f"Model was saved to: {model_path}")
                else:
                    click.echo(f"Error while saving model with path: {model_path}")
        return {"clf": clf}
