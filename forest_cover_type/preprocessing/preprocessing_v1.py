"""
    preparing the data: cleaning, integration, reduction, transformation

    handling the missing values
    splitting the dataset
    encoding the categorical data
    feature finding and scaling
"""
import os, sys
import numpy as np
import pandas as pd
from loguru import logger  # type:ignore
from sklearn.preprocessing import normalize
from ..package_two import module_two


def run(settings):
    logger.info("preprocessing")
    path = os.path.join(settings.dataset_path, "train.csv")
    if os.path.isfile(path):
        df_train = pd.read_csv(path)
    else:
        raise FileNotFoundError(path)
    path = os.path.join(settings.dataset_path, "test.csv")
    if os.path.isfile(path):
        df_test = pd.read_csv(path)
    else:
        raise FileNotFoundError(path)
    print("df_train.shape, df_test.shape")
    print(df_train.shape, df_test.shape)
    assert df_train.isna().sum().sum() == 0
    assert df_test.isna().sum().sum() == 0
    # target Cover_Type
    # train.Cover_Type.unique() # 1..7
    cols_to_normalize = [
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    df_train[cols_to_normalize] = normalize(df_train[cols_to_normalize])
    df_test[cols_to_normalize] = normalize(df_test[cols_to_normalize])

    feature_cols = [col for col in df_train.columns if col not in ["Cover_Type", "Id"]]
    feature_cols.append("binned_elevation")
    feature_cols.append("Horizontal_Distance_To_Roadways_Log")
    feature_cols.append("Soil_Type12_32")
    feature_cols.append("Soil_Type23_22_32_33")
    df_train["binned_elevation"] = [np.floor(v / 50.0) for v in df_train["Elevation"]]
    df_test["binned_elevation"] = [np.floor(v / 50.0) for v in df_test["Elevation"]]
    df_train["Horizontal_Distance_To_Roadways_Log"] = [
        np.log(v + 1) for v in df_train["Horizontal_Distance_To_Roadways"]
    ]
    df_test["Horizontal_Distance_To_Roadways_Log"] = [
        np.log(v + 1) for v in df_test["Horizontal_Distance_To_Roadways"]
    ]
    df_train["Soil_Type12_32"] = df_train["Soil_Type32"] + df_train["Soil_Type12"]
    df_test["Soil_Type12_32"] = df_test["Soil_Type32"] + df_test["Soil_Type12"]
    df_train["Soil_Type23_22_32_33"] = (
        df_train["Soil_Type23"] + df_train["Soil_Type22"] + df_train["Soil_Type32"] + df_train["Soil_Type33"]
    )
    df_test["Soil_Type23_22_32_33"] = (
        df_test["Soil_Type23"] + df_test["Soil_Type22"] + df_test["Soil_Type32"] + df_test["Soil_Type33"]
    )

    df_train_1_2 = df_train[(df_train["Cover_Type"] <= 2)]
    df_train_3_4_6 = df_train[(df_train["Cover_Type"].isin([3, 4, 6]))]

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    X_train_1_2 = df_train_1_2[feature_cols]
    X_train_3_4_6 = df_train_3_4_6[feature_cols]

    y = df_train["Cover_Type"]
    y_1_2 = df_train_1_2["Cover_Type"]
    y_3_4_6 = df_train_3_4_6["Cover_Type"]

    # test_ids = df_test["Id"]
    return {
        "train_dataframes": [(X_train, y), (X_train_1_2, y_1_2), (X_train_3_4_6, y_3_4_6)],
        "test_dataframe": X_test,
        "sub_dataframe": df_test["Id"],
    }
