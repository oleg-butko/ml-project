import sys
import numpy as np
import pandas as pd
from loguru import logger  # type:ignore

from .. import utils

global glob
glob = utils.dotdict({})


def largest_index(inlist):
    largest = -1
    largest_index = 0
    for i in range(len(inlist)):
        item = inlist[i]
        if item > largest:
            largest = item
            largest_index = i
    return largest_index


def run(settings, classifiers, df, sub_df=None):
    logger.info(f"predict, df.shape: {df.shape}")
    clf = classifiers["clf"]
    # from forest_cover_type.utils import dotdict
    # s = dotdict(sys.modules["forest_cover_type"].runner.g_settings)
    # s.vars.keys()
    # s.vars.X_train.shape (2451, 54)
    # s.vars.df.shape
    # settings.vars.df = df
    # sys.exit()
    y_pred = clf.predict_proba(df)
    if settings.use_booster:
        clf_1_2 = classifiers["clf_1_2"]
        clf_3_4_6 = classifiers["clf_3_4_6"]
        y_pred_1_2 = clf_1_2.predict_proba(df)
        y_pred_3_4_6 = clf_3_4_6.predict_proba(df)
        y_pred[:, 0:2] += y_pred_1_2[:, 0:2] / settings.coef_1
        y_pred[:, 2:4] += y_pred_3_4_6[:, 0:2] / settings.coef_2
        y_pred[:, 5] += y_pred_3_4_6[:, 2] / settings.coef_3
    y_pred = y_pred.argsort(axis=1)[:, -1] + 1
    if sub_df is not None:
        predictions_df = pd.DataFrame(np.column_stack((sub_df.values, y_pred)), columns=["Id", "Cover_Type"])
    else:
        predictions_df = pd.DataFrame((y_pred), columns=["Cover_Type"])
    return predictions_df


def with_new_labels(settings, classifiers, processed):
    # sub_df = processed["sub_dataframe"]
    train_df, y_true = processed["train_dataframes"][0]
    if not processed.get("test_dataframe_pl_exists", False):
        test_df = processed["test_dataframe"]
    else:
        test_df = processed["test_dataframe_pl"]
    settings.vars.test_df = test_df
    logger.info("with_new_labels")
    logger.info(f"test_df.shape: {test_df.shape}")
    logger.info(f"train_df.shape: {train_df.shape}")

    clf = classifiers["clf"]
    clf_1_2 = classifiers["clf_1_2"]
    clf_3_4_6 = classifiers["clf_3_4_6"]

    y_pred = clf.predict_proba(test_df)
    y_pred_1_2 = clf_1_2.predict_proba(test_df)
    y_pred_3_4_6 = clf_3_4_6.predict_proba(test_df)

    y_pred[:, 0:2] += y_pred_1_2[:, 0:2] / settings.coef_1
    y_pred[:, 2:4] += y_pred_3_4_6[:, 0:2] / settings.coef_2
    y_pred[:, 5] += y_pred_3_4_6[:, 2] / settings.coef_3
    # y_pred_t (565892, 7)
    y_pred_t = y_pred.argsort(axis=1)[:, -1] + 1  # index + 1
    y_pred_t = pd.Series(y_pred_t, name="Cover_Type")

    # more_than099 = np.where(y_pred > 0.99)[0]
    more_than = np.where(y_pred > (np.max(y_pred) - 0.03))[0]
    logger.info(f"more_than.shape: {more_than.shape}")
    n_pseudolabels = min(25000, more_than.shape[0])

    idx = np.random.choice(more_than, size=n_pseudolabels, replace=False)  # (461138,)
    settings.vars.idx = idx
    settings.vars.y_pred = y_pred
    settings.vars.y_pred_t = y_pred_t

    # print(test_df.iloc[idx].shape)
    # print(y_pred_t[idx].shape)
    # v.test_df.iloc[v.idx].shape (10, 58)
    test_df_small_part = test_df.iloc[idx]
    train_df = train_df.append(test_df_small_part, ignore_index=True)
    y_true = y_true.append(y_pred_t[idx], ignore_index=True)

    print("np.unique(y_true):", np.unique(y_true, return_counts=True))
    # logger.info(f"train_df.shape: {train_df.shape}")
    # v.test_df.iloc[v.idx].shape (10, 58)
    # v.test_df.shape (565892, 58)
    # v.test_df.loc[~v.test_df.index.isin(v.idx)].shape (565882, 58)
    test_df_the_rest = test_df.loc[~test_df.index.isin(idx)]
    processed["test_dataframe_pl"] = test_df_the_rest
    processed["test_dataframe_pl_exists"] = True

    train_df["Cover_Type"] = y_true
    X_train_1_2 = train_df[(train_df["Cover_Type"] <= 2)]

    y_1_2 = X_train_1_2["Cover_Type"]
    X_train_1_2 = X_train_1_2.drop("Cover_Type", axis=1)
    train_df.drop("Cover_Type", axis=1, inplace=True)
    # X_train_1_2, y_1_2 = processed["train_dataframes"][1]
    X_train_3_4_6, y_3_4_6 = processed["train_dataframes"][2]
    processed["train_dataframes"] = [
        (train_df, y_true),
        (X_train_1_2, y_1_2),
        (X_train_3_4_6, y_3_4_6),
    ]
    # from forest_cover_type.utils import dotdict
    # v = dotdict(sys.modules["forest_cover_type"].runner.g_settings.vars)
    # v.keys()
    # v.X_train.shape
    # sys.exit()
    # if sub_df is not None:
    #     predictions_df = pd.DataFrame(np.column_stack((sub_df.values, y_pred)), columns=["Id", "Cover_Type"])
    # else:
    #     predictions_df = pd.DataFrame((y_pred), columns=["Cover_Type"])
    return processed
