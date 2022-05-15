import sys
from joblib import dump, load
from loguru import logger  # type:ignore
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from ..report import kaggle_utils

use_mlflow = True
try:
    import mlflow
except ModuleNotFoundError:
    use_mlflow = False


def roc_auc_scorer(clf, X, y):
    y_true = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    y_pred = clf.predict_proba(X)
    return roc_auc_score(y_true, y_pred, multi_class="ovr")


def get_scorer_for(cls_label):
    assert cls_label >= 1 and cls_label <= 7

    def sum_err_for_cls_label(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        # normalize
        cm2 = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # sum percents of predictions errors
        err_pred_sum = np.sum(cm2[:, cls_label - 1]) - cm2[cls_label - 1, cls_label - 1]
        # print(f"cls_label: {cls_label}  err_pred_sum: {err_pred_sum}")
        return err_pred_sum

    return sum_err_for_cls_label


#
# %run -m forest_cover_type.runner -t cfg/kfold.ini
#
def kfold(settings, processed, run_n=None):
    dataframes = processed["train_dataframes"]
    assert len(dataframes) > 0
    assert run_n is not None
    random_state = settings.SEED
    X, y = dataframes[0]
    # settings.X = X
    # sys.exit()
    X, y = X.values, y.values
    run_name = f"{run_n} {settings.runs[run_n].classifier}"
    logger.info(f"{run_name} kfold, X.shape: {X.shape}, n_splits: {settings.n_splits}")
    skf = StratifiedKFold(n_splits=settings.n_splits, shuffle=True, random_state=random_state)
    # for train_index, test_index in skf.split(X, y):
    #     print("TRAIN:", len(train_index), "TEST:", len(test_index))
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     print("np.unique(y_test):", np.unique(y_test))

    # del few params which are not for classifier
    classifier = settings.runs[run_n].classifier
    del settings.runs[run_n]["classifier"]
    create_submission_file = settings.runs[run_n].create_submission_file  # True or None
    if create_submission_file:
        del settings.runs[run_n]["create_submission_file"]

    logger.info(f"  params: {settings.runs[run_n]}")
    if classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier(**(settings.runs[run_n]), random_state=random_state)
    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier(**(settings.runs[run_n]), random_state=random_state)
    else:
        logger.error(f"Invalid classifier in: {run_n}")
        raise ValueError

    if create_submission_file:
        clf.fit(X, y)
        test_df = processed["test_dataframe"]
        y_pred = clf.predict_proba(test_df.values)
        y_pred = y_pred.argsort(axis=1)[:, -1] + 1
        sub_df = processed["sub_dataframe"]
        df = pd.DataFrame(np.column_stack((sub_df.values, y_pred)), columns=["Id", "Cover_Type"])
        filename = settings.submission_fn_tmpl % run_name
        df.to_csv(filename, index=False)
        settings.create_submission_file = False
        logger.info(f"Submission file '{filename}' was created.")
        return

    if use_mlflow:
        for k, v in settings.runs[run_n].items():
            mlflow.log_param(k, v)
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    scoring = {
        "f1_macro": "f1_macro",
        "balanced_acc": "balanced_accuracy",
        "neg_log_loss": "neg_log_loss",
        "roc_auc": roc_auc_scorer,
    }
    for i in range(1, 8):
        scoring[f"label_{i}_err"] = get_scorer_for(i)
    # KFold
    cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=True)
    # print(cv_results.keys())
    # 'fit_time', 'score_time', 'test_roc_auc', 'train_roc_auc', 'test_balanced_acc', 'train_balanced_acc', 'test_neg_log_loss', 'train_neg_log_loss', 'test_label_1_err', 'train_label_1_err', 'test_label_2_err', 'train_label_2_err', 'test_label_3_err', 'train_label_3_err'])
    if use_mlflow:
        mlflow.log_metric("train_f1_ma", cv_results["train_f1_macro"].mean())
        mlflow.log_metric("test_f1_ma", cv_results["test_f1_macro"].mean())
        mlflow.log_metric("train_roc_auc", cv_results["train_roc_auc"].mean())
        mlflow.log_metric("test_roc_auc", cv_results["test_roc_auc"].mean())
        mlflow.log_metric("train_bal_acc", cv_results["train_balanced_acc"].mean())
        mlflow.log_metric("test_bal_acc", cv_results["test_balanced_acc"].mean())
        mlflow.log_metric("train_log_loss", cv_results["train_neg_log_loss"].mean())
        mlflow.log_metric("test_log_loss", cv_results["test_neg_log_loss"].mean())
        for i in range(1, 8):
            mlflow.log_metric(f"train_lab_{i}_err", cv_results[f"train_label_{i}_err"].mean())
            mlflow.log_metric(f"test_lab_{i}_err", cv_results[f"test_label_{i}_err"].mean())


def xgb_1(settings, processed):
    X_train, y_train = processed["train_dataframes"][0]
    y_train = y_train - 1
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # params = {
    #     "objective": "reg:linear",
    #     "max_depth": 2,
    #     "learning_rate": 0.1,
    #     "min_child_weight": 3,
    #     "colsample_bytree": 0.7,
    #     "subsample": 0.8,
    #     "gamma": 0,
    #     "alpha": 1,
    # }
    params = {
        "booster": "gbtree",
        "objective": "multi:softmax",
        "num_class": 7,
        "verbosity": 1,
    }
    # trees = 100
    # trees = 100 # 0.73681
    # trees = 1000 # 0.75545
    # logger.info(f"xgb.cv, X_train.shape: {X_train.shape}")
    # cv = xgb.cv(
    #     params,
    #     dtrain,
    #     metrics=("merror"),
    #     verbose_eval=False,
    #     nfold=4,
    #     show_stdv=False,
    #     stratified=True,
    #     num_boost_round=trees,
    #     seed=settings.random_state,
    # )
    logger.info("xgb.train")
    # bst = xgb.train(params, dtrain, num_boost_round=cv["test-merror-mean"].argmin())
    bst = xgb.train(params, dtrain, num_boost_round=1000)  # 0.75858
    y_pred_train = bst.predict(dtrain)
    settings.vars.y_pred_train = y_pred_train
    settings.vars.y_true = y_train
    # average : {'micro', 'macro', 'samples','weighted', 'binary'}
    f1_w = f1_score(y_true=y_train, y_pred=y_pred_train, average="weighted")
    logger.info(f"f1_w: {f1_w}")
    f1_ma = f1_score(y_true=y_train, y_pred=y_pred_train, average="macro")
    logger.info(f"f1_ma: {f1_ma}")
    settings.vars.cm = confusion_matrix(y_train, y_pred_train)

    X_test = processed["test_dataframe"]
    dtest = xgb.DMatrix(X_test)
    y_pred_test = bst.predict(dtest).astype(int)
    y_pred_test = y_pred_test
    # y_pred_test = y_pred_test + 1
    settings.vars.y_pred_test = y_pred_test
    sub_df = processed["sub_dataframe"]
    predictions_df = pd.DataFrame(
        np.column_stack((sub_df.values, y_pred_test)), columns=["Id", "Cover_Type"], dtype=int
    )
    settings.vars.predictions_df = predictions_df
    predictions_df.to_csv(settings.submission_fn, index=False)
    logger.info(f"Created {settings.submission_fn}.")
    kaggle_utils.upload_and_get_score(settings)

    # from forest_cover_type.utils import dotdict
    # s = dotdict(sys.modules["forest_cover_type"].runner.g_settings)
    # s.vars.keys()
    # s.vars.pred_train.shape (15120,)

    # average : {'micro', 'macro', 'samples','weighted', 'binary'} or None, \
    # acc_on_train = f1_score(g_settings.vars.y, predictions_df).round(5)
    # logger.info(f"acc_on_train: {acc_on_train}")


def run(settings, dataframes):
    assert len(dataframes) > 0
    clf = ensemble.ExtraTreesClassifier(
        n_estimators=settings.clf_n_estimators,
        max_depth=settings.max_depth,
        n_jobs=settings.n_jobs,
        random_state=settings.random_state,
        class_weight={1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
    )
    # ccp_alpha=0.001 0.74942, ccp_alpha=0.0001 0.80065
    # class_weight={1: 10, 2: 10, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1} 0.83067 -> 0.82956
    # class_weight={1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1} 0.83067 -> 0.83044
    if not settings.use_booster:
        if settings.load_if_exists and settings.model_path.is_file() and settings.model_path.exists():
            clf = load(settings.model_path)
            logger.info(f"Loaded model from path: {settings.model_path}")

    X_train, y = dataframes[0]
    logger.info(f"clf.fit(X_train, y), X_train.shape: {X_train.shape}")
    # from forest_cover_type.utils import dotdict
    # s = dotdict(sys.modules["forest_cover_type"].runner.g_settings)
    # s.vars.keys()
    # s.vars.X_train_1.shape (2451, 58)
    # s.vars.test_df.shape
    # settings.vars.X_train_1 = X_train
    assert X_train.shape[1] == 58
    clf.fit(X_train, y)

    if settings.use_booster:
        assert len(dataframes) == 3
        clf_1_2 = ensemble.ExtraTreesClassifier(
            n_estimators=settings.booster_n_estimators_1,
            n_jobs=settings.n_jobs,
            random_state=settings.random_state,
        )
        X_train_1_2, y_1_2 = dataframes[1]
        logger.info(f"clf_1_2.fit(X_train_1_2, y_1_2), X_train_1_2.shape: {X_train_1_2.shape}")
        clf_1_2.fit(X_train_1_2, y_1_2)
        clf_3_4_6 = ensemble.ExtraTreesClassifier(
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
