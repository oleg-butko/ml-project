from sklearn import ensemble
from forest_cover_type import settings


def run(dataframes):
    assert len(dataframes) == 3
    clf = ensemble.ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=settings.SEED)
    X_train, y = dataframes[0]
    clf.fit(X_train, y)
    clf_1_2 = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=settings.SEED)
    X_train_1_2, y_1_2 = dataframes[1]
    clf_1_2.fit(X_train_1_2, y_1_2)
    clf_3_4_6 = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=settings.SEED)
    X_train_3_4_6, y_3_4_6 = dataframes[2]
    clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
    return {"clf": clf, "clf_1_2": clf_1_2, "clf_3_4_6": clf_3_4_6}
