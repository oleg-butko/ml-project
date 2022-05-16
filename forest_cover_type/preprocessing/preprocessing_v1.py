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
from sklearn.ensemble import RandomForestClassifier

# https://github.com/scikit-learn-contrib/boruta_py
boruta_not_found = False
try:
    from boruta import BorutaPy
except ModuleNotFoundError:
    boruta_not_found = True


def load_data(settings):
    path = os.path.join(settings.dataset_path, "train.csv")
    if os.path.isfile(path):
        df_train = pd.read_csv(path)
        logger.info(f"read_csv: {path}")
    else:
        raise FileNotFoundError(path)
    if settings.load_test_csv:
        path = os.path.join(settings.dataset_path, "test.csv")
        if os.path.isfile(path):
            df_test = pd.read_csv(path)
            logger.info(f"read_csv: {path}")
        else:
            raise FileNotFoundError(path)
    else:
        df_test = pd.DataFrame(columns=["Id"])
    return df_train, df_test


def load_new_labels(settings):
    path = os.path.join(settings.dataset_path, "new_big_train.csv")
    if os.path.isfile(path):
        df_big_train = pd.read_csv(path)
        logger.info(f"read_csv: {path}")
    else:
        raise FileNotFoundError(path)
    return df_big_train


def fe_none(settings, df_train, df_test):
    X_train = df_train.drop(["Cover_Type", "Id"], axis=1)
    y = df_train["Cover_Type"]
    return {
        "train_dataframes": [(X_train, y)],
        "test_dataframe": df_test.drop("Id", axis=1),
        "sub_dataframe": df_test["Id"],
    }


def fe_1(settings, df_train, df_test):
    if boruta_not_found:
        logger.warning("boruta not found, using fe_none instead of fe_1")
        return fe_none(settings, df_train, df_test)

    feat_selector = BorutaPy(
        RandomForestClassifier(
            max_depth=None,
            ccp_alpha=0.0002,
            class_weight={1: 11, 2: 11, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        ),
        n_estimators="auto",
        max_iter=50,
        random_state=settings.random_state,
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


def transform_1(df):
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
    # df[cols_to_normalize] = StandardScaler().fit_transform(df[cols_to_normalize]) # 0.79067
    # normalize: 0.82769, no normalize: 0.81461
    df[cols_to_normalize] = normalize(df[cols_to_normalize])
    df["binned_elevation"] = [np.floor(v / 50.0) for v in df["Elevation"]]
    df["Horizontal_Distance_To_Roadways_Log"] = [np.log(v + 1) for v in df["Horizontal_Distance_To_Roadways"]]
    df["Soil_Type12_32"] = df["Soil_Type32"] + df["Soil_Type12"]
    df["Soil_Type23_22_32_33"] = df["Soil_Type23"] + df["Soil_Type22"] + df["Soil_Type32"] + df["Soil_Type33"]
    return df


def fe_2(settings, df_train, df_test):
    feature_cols = [col for col in df_train.columns if col not in ["Cover_Type", "Id"]]
    feature_cols.append("binned_elevation")
    feature_cols.append("Horizontal_Distance_To_Roadways_Log")
    feature_cols.append("Soil_Type12_32")
    feature_cols.append("Soil_Type23_22_32_33")
    df_train = transform_1(df_train)
    df_train_1_2 = df_train[(df_train["Cover_Type"] <= 2)]
    df_train_3_4_6 = df_train[(df_train["Cover_Type"].isin([3, 4, 6]))]
    # print("df_train_1_2", df_train_1_2["Cover_Type"].value_counts().to_dict())
    # print("df_train_3_4_6", df_train_3_4_6["Cover_Type"].value_counts().to_dict())
    # df_train_1_2 {2: 2160, 1: 2160}
    # df_train_3_4_6 {3: 2160, 6: 2160, 4: 2160}

    X_train = df_train[feature_cols]
    settings.vars.df_train = df_train
    if df_test.shape[0] > 0:
        df_test = transform_1(df_test)
        X_test = df_test[feature_cols]
        settings.vars.df_test = df_test
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
    if settings.use_pl:
        df_big_train = load_new_labels(settings)
    logger.info(f"Loaded: df_train, df_test: {df_train.shape} {df_test.shape}")
    if settings.feature_engineering == "fe_1":
        result = fe_1(settings, df_train, df_test)
    elif settings.feature_engineering == "fe_2":
        result = fe_2(settings, df_train, df_test)
    else:
        result = fe_none(settings, df_train, df_test)
        # logger.error("Invalid feature_engineering value in config")
        # raise ValueError
    X_train = result["train_dataframes"][0][0]
    X_test = result["test_dataframe"]
    logger.info(f"After feature_engineering: {settings.feature_engineering}")
    logger.info(f"X_train.shape: {X_train.shape}")
    logger.info(f"X_test.shape: {X_test.shape}")
    # assert df_train.isna().sum().sum() == 0
    # assert X_train.isna().sum().sum() == 0
    if settings.use_pl:
        result["df_big_train"] = df_big_train
    return result
