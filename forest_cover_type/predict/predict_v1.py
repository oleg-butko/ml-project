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
    # s = dotdict(sys.modules["forest_cover_type"].runner.settings_obj)
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
        y_pred[:, 0:2] += y_pred_1_2[:, 0:2] / 1.3
        y_pred[:, 2:4] += y_pred_3_4_6[:, 0:2] / 3.9
        y_pred[:, 5] += y_pred_3_4_6[:, 2] / 3.6
    y_pred = y_pred.argsort(axis=1)[:, -1] + 1
    if sub_df is not None:
        predictions_df = pd.DataFrame(np.column_stack((sub_df.values, y_pred)), columns=["Id", "Cover_Type"])
    else:
        predictions_df = pd.DataFrame((y_pred), columns=["Cover_Type"])
    return predictions_df
