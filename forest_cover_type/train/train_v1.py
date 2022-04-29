import sys, configparser, distutils
from sklearn import ensemble


def run(settings, dataframes):
    assert len(dataframes) == 3
    # defaults, can be overwritten by train.ini
    n_jobs = -1
    clf_n_estimators = 100
    use_booster = False
    booster_n_estimators_1 = 200
    booster_n_estimators_2 = 200
    if settings.train_cfg is not None:
        config = configparser.ConfigParser()
        config.read(settings.train_cfg, encoding="utf-8")
        n_jobs = int(config.get("Common", "n_jobs", fallback=n_jobs))
        clf_n_estimators = int(config.get("ExtraTreesClassifier", "n_estimators", fallback=clf_n_estimators))
        use_booster = distutils.util.strtobool(config.get("Booster", "enable", fallback=use_booster))
        # send this option to the predict step
        settings.use_booster = use_booster
        booster_n_estimators_1 = int(config.get("Booster", "n_estimators_1", fallback=booster_n_estimators_1))
        booster_n_estimators_2 = int(config.get("Booster", "n_estimators_2", fallback=booster_n_estimators_2))

    clf = ensemble.ExtraTreesClassifier(n_estimators=100, n_jobs=n_jobs, random_state=settings.SEED)
    X_train, y = dataframes[0]
    clf.fit(X_train, y)
    if use_booster:
        clf_1_2 = ensemble.RandomForestClassifier(
            n_estimators=booster_n_estimators_1, n_jobs=n_jobs, random_state=settings.SEED
        )
        X_train_1_2, y_1_2 = dataframes[1]
        clf_1_2.fit(X_train_1_2, y_1_2)
        clf_3_4_6 = ensemble.RandomForestClassifier(
            n_estimators=booster_n_estimators_2, n_jobs=n_jobs, random_state=settings.SEED
        )
        X_train_3_4_6, y_3_4_6 = dataframes[2]
        clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
        return {"clf": clf, "clf_1_2": clf_1_2, "clf_3_4_6": clf_3_4_6}
    else:
        return {"clf": clf}
