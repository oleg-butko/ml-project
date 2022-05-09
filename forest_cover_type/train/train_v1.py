import sys
from joblib import dump, load
from loguru import logger  # type:ignore
import numpy as np
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.preprocessing import OneHotEncoder


def roc_auc_scorer(clf, X, y):
    y_true = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    y_pred = clf.predict_proba(X)
    return roc_auc_score(y_true, y_pred, multi_class="ovr")


def get_scorer_for(label):
    def sum_err_for_label(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        # normalaze
        cm2 = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # sum percents of predictions errors
        err_pred_sum = np.sum(cm2[:, label - 1]) - cm2[label - 1, label - 1]
        # print(f"label: {label}  err_pred_sum: {err_pred_sum}")
        return err_pred_sum

    return sum_err_for_label


def kfold(settings, dataframes):
    assert len(dataframes) > 0
    random_state = settings.SEED
    X, y = dataframes[0]
    X, y = X.values, y.values
    logger.info(f"kfold, X.shape: {X.shape}, n_splits: {settings.n_splits}")
    skf = StratifiedKFold(n_splits=settings.n_splits, shuffle=True, random_state=random_state)
    # for train_index, test_index in skf.split(X, y):
    #     print("TRAIN:", len(train_index), "TEST:", len(test_index))
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     print("np.unique(y_test):", np.unique(y_test))
    tree = DecisionTreeClassifier(max_depth=10, random_state=random_state)
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    scoring = {
        "roc_auc": roc_auc_scorer,
        "balanced_acc": "balanced_accuracy",
        "neg_log_loss": "neg_log_loss",
        "label_1_err": get_scorer_for(1),
        "label_2_err": get_scorer_for(2),
        "label_3_err": get_scorer_for(3),
    }
    cv_results = cross_validate(tree, X, y, cv=skf, scoring=scoring, return_train_score=True)
    # print(cv_results.keys())
    # 'fit_time', 'score_time', 'test_roc_auc', 'train_roc_auc', 'test_balanced_acc', 'train_balanced_acc', 'test_neg_log_loss', 'train_neg_log_loss', 'test_label_1_err', 'train_label_1_err', 'test_label_2_err', 'train_label_2_err', 'test_label_3_err', 'train_label_3_err'])
    print("train_label_1_err", cv_results["train_label_1_err"])
    print("test_label_1_err", cv_results["test_label_1_err"])
    print("train_balanced_acc", cv_results["train_balanced_acc"])
    print("test_balanced_acc", cv_results["test_balanced_acc"])
    print("train_roc_auc", cv_results["train_roc_auc"])
    print("test_roc_auc", cv_results["test_roc_auc"])


def run(settings, dataframes):
    assert len(dataframes) == 3
    clf = ensemble.ExtraTreesClassifier(
        n_estimators=settings.clf_n_estimators,
        max_depth=settings.max_depth,
        n_jobs=settings.n_jobs,
        random_state=settings.random_state,
    )
    if not settings.use_booster:
        if settings.load_if_exists and settings.model_path.is_file() and settings.model_path.exists():
            clf = load(settings.model_path)
            logger.info(f"Loaded model from path: {settings.model_path}")

    X_train, y = dataframes[0]
    logger.info(f"clf.fit(X_train, y), X_train.shape: {X_train.shape}")
    print("clf_n_estimators:", settings.clf_n_estimators)
    print("max_depth:", settings.max_depth)
    clf.fit(X_train, y)

    if settings.use_booster:
        clf_1_2 = ensemble.RandomForestClassifier(
            n_estimators=settings.booster_n_estimators_1,
            n_jobs=settings.n_jobs,
            random_state=settings.random_state,
        )
        X_train_1_2, y_1_2 = dataframes[1]
        logger.info(f"clf_1_2.fit(X_train_1_2, y_1_2), X_train_1_2.shape: {X_train_1_2.shape}")
        clf_1_2.fit(X_train_1_2, y_1_2)
        clf_3_4_6 = ensemble.RandomForestClassifier(
            n_estimators=settings.booster_n_estimators_2,
            n_jobs=settings.n_jobs,
            random_state=settings.random_state,
        )
        X_train_3_4_6, y_3_4_6 = dataframes[2]
        logger.info(f"clf_3_4_6.fit(X_train_3_4_6, y_1_2), X_train_3_4_6.shape: {X_train_3_4_6.shape}")
        clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
        return {"clf": clf, "clf_1_2": clf_1_2, "clf_3_4_6": clf_3_4_6}
    else:
        if settings.save_if_not_exists:
            if settings.model_path.exists():
                pass
            else:
                dump(clf, settings.model_path)
                if settings.model_path.is_file() and settings.model_path.exists():
                    logger.info(f"Model was saved to: {settings.model_path}")
                else:
                    logger.error(f"Error while saving model with path: {settings.model_path}")
        return {"clf": clf}
