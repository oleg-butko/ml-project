import numpy as np
import pandas as pd


def largest_index(inlist):
    largest = -1
    largest_index = 0
    for i in range(len(inlist)):
        item = inlist[i]
        if item > largest:
            largest = item
            largest_index = i
    return largest_index


def run(settings, classifiers, test_df, sub_df):
    clf = classifiers["clf"]
    y_pred = clf.predict_proba(test_df)
    if settings.use_booster:
        clf_1_2 = classifiers["clf_1_2"]
        clf_3_4_6 = classifiers["clf_3_4_6"]
        y_pred_1_2 = clf_1_2.predict_proba(test_df)
        y_pred_3_4_6 = clf_3_4_6.predict_proba(test_df)
        y_pred[:, 0:2] += y_pred_1_2[:, 0:2]
        y_pred[:, 2:4] += y_pred_3_4_6[:, 0:2]
        y_pred[:, 5] += y_pred_3_4_6[:, 2]
    y_pred = y_pred.argsort(axis=1)[:, -1] + 1
    predictions_df = pd.DataFrame(np.column_stack((sub_df.values, y_pred)), columns=["Id", "Cover_Type"])
    return predictions_df
