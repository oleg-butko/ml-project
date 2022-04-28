"""
    preparing the data: cleaning, integration, reduction, transformation

    handling the missing values
    splitting the dataset
    encoding the categorical data
    feature finding and scaling
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from forest_cover_type import settings
from forest_cover_type.package_two import module_two


def run():
    if os.path.isfile(settings.PATH + "train.csv"):
        df_train = pd.read_csv(settings.PATH + "train.csv")
        df_test = pd.read_csv(settings.PATH + "test.csv")
    elif os.path.isfile(settings.PATH_2 + "train.csv"):
        df_train = pd.read_csv(settings.PATH_2 + "train.csv")
        df_test = pd.read_csv(settings.PATH_2 + "test.csv")
    print(df_train.shape)
    print(df_test.shape)
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
