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
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# from boruta import BorutaPy


# from ..package_two import module_two


def load_data(settings):
    path = os.path.join(settings.dataset_path, "train.csv")
    if os.path.isfile(path):
        df_train = pd.read_csv(path)
        logger.info(f"read_csv: {path}")
    else:
        raise FileNotFoundError(path)
    if settings.create_submission_file:
        path = os.path.join(settings.dataset_path, "test.csv")
        if os.path.isfile(path):
            df_test = pd.read_csv(path)
            logger.info(f"read_csv: {path}")
        else:
            raise FileNotFoundError(path)
        assert df_test.isna().sum().sum() == 0
    else:
        df_test = pd.DataFrame(columns=["Id"])
    return df_train, df_test


def fe_1(settings, df_train, df_test):
    return
    feat_selector = BorutaPy(
        RandomForestClassifier(
            max_depth=None,
            ccp_alpha=0.0002,
            class_weight={1: 11, 2: 11, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        ),
        n_estimators="auto",
        max_iter=50,
        random_state=settings.SEED,
        verbose=1,
    )
    X_train = df_train.drop(["Cover_Type", "Id"], axis=1)
    y = df_train["Cover_Type"]
    feat_selector.fit(X_train.values, y.values)
    X_train_filtered = feat_selector.transform(X_train.values)
    X_train = pd.DataFrame(X_train_filtered)
    logger.info(f"After boruta X_train.shape {X_train.shape}")
    if df_test.shape[0] > 0:
        index = df_test.Id
        df_test_filtered = feat_selector.transform(df_test.drop("Id", axis=1).values)
        df_test = pd.DataFrame(df_test_filtered, index=index)
        logger.info(f"After boruta df_test.shape {df_test.shape}")
        df_test.reset_index(inplace=True)
    return {
        "train_dataframes": [(X_train, y)],
        "test_dataframe": df_test.drop("Id", axis=1),
        "sub_dataframe": df_test["Id"],
    }


def fe_2(settings, df_train, df_test):
    # target Cover_Type
    # train.Cover_Type.unique() # 1..7
    # {2: 860, 5: 641, 1: 499, 6: 179, 7: 125, 3: 107, 4: 40}
    # {5: 2160, 2: 2160, 1: 2160, 7: 2160, 3: 2160, 6: 2160, 4: 2160}
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
    feature_cols = [col for col in df_train.columns if col not in ["Cover_Type", "Id"]]
    feature_cols.append("binned_elevation")
    feature_cols.append("Horizontal_Distance_To_Roadways_Log")
    feature_cols.append("Soil_Type12_32")
    feature_cols.append("Soil_Type23_22_32_33")
    df_train["binned_elevation"] = [np.floor(v / 50.0) for v in df_train["Elevation"]]
    df_train["Horizontal_Distance_To_Roadways_Log"] = [
        np.log(v + 1) for v in df_train["Horizontal_Distance_To_Roadways"]
    ]
    df_train["Soil_Type12_32"] = df_train["Soil_Type32"] + df_train["Soil_Type12"]
    df_train["Soil_Type23_22_32_33"] = (
        df_train["Soil_Type23"] + df_train["Soil_Type22"] + df_train["Soil_Type32"] + df_train["Soil_Type33"]
    )

    df_train_1_2 = df_train[(df_train["Cover_Type"] <= 2)]
    df_train_3_4_6 = df_train[(df_train["Cover_Type"].isin([3, 4, 6]))]

    X_train = df_train[feature_cols]

    if df_test.shape[0] > 0:
        df_test[cols_to_normalize] = normalize(df_test[cols_to_normalize])
        df_test["binned_elevation"] = [np.floor(v / 50.0) for v in df_test["Elevation"]]
        df_test["Horizontal_Distance_To_Roadways_Log"] = [
            np.log(v + 1) for v in df_test["Horizontal_Distance_To_Roadways"]
        ]
        df_test["Soil_Type12_32"] = df_test["Soil_Type32"] + df_test["Soil_Type12"]
        df_test["Soil_Type23_22_32_33"] = (
            df_test["Soil_Type23"] + df_test["Soil_Type22"] + df_test["Soil_Type32"] + df_test["Soil_Type33"]
        )
        X_test = df_test[feature_cols]
    else:
        X_test = df_test

    X_train_1_2 = df_train_1_2[feature_cols]
    X_train_3_4_6 = df_train_3_4_6[feature_cols]

    y = df_train["Cover_Type"]
    y_1_2 = df_train_1_2["Cover_Type"]
    y_3_4_6 = df_train_3_4_6["Cover_Type"]

    return {
        "train_dataframes": [(X_train, y), (X_train_1_2, y_1_2), (X_train_3_4_6, y_3_4_6)],
        "test_dataframe": X_test,
        "sub_dataframe": df_test["Id"],
    }


def run(settings):
    df_train, df_test = load_data(settings)
    logger.info(f"feature_engineering: {settings.feature_engineering}")
    logger.info(f"df_train, df_test: {df_train.shape} {df_test.shape}")
    if settings.feature_engineering == "fe_1":
        result = fe_1(settings, df_train, df_test)
    elif settings.feature_engineering == "fe_2":
        result = fe_2(settings, df_train, df_test)
    else:
        logger.error("Invalid feature_engineering value in config")
        raise ValueError
    X_train = result["train_dataframes"][0][0]
    logger.info(f"X_train.shape: {X_train.shape}")
    assert df_train.isna().sum().sum() == 0
    assert X_train.isna().sum().sum() == 0
    return result
