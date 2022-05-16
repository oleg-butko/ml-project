import sys, os, traceback, warnings, logging, time
from pathlib import Path
import click
from loguru import logger  # type:ignore
from sklearn.metrics import accuracy_score, f1_score


# Hide warnings from sklearn about deprecation that mlflow shows
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# mlflow is not available from qtconsole (for debugging): %run -m forest_cover_type.runner
use_mlflow = True
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    use_mlflow = False


from . import settings as settings_py
from . import __version__
from . import utils
from .preprocessing import preprocessing_v1
from .train import train_v1
from .predict import predict_v1
from .report import kaggle_utils

# Correct calls from qtconsole after kernel reload:
# %run -m forest_cover_type -a
# r1 (%run -m forest_cover_type)
# %run -m forest_cover_type -d data/only2krows -t cfg/kfold.ini

# sys.modules["forest_cover_type"].runner.g_settings.vars.keys()
# from forest_cover_type.utils import dotdict
# v = dotdict(sys.modules["forest_cover_type"].runner.g_settings.vars)
# v.keys() -- still keeps old values after new run
# v.X_train.shape (2451, 54)
# settings.vars.df = df
# {5: 2160, 2: 2160, 1: 2160, 7: 2160, 3: 2160, 6: 2160, 4: 2160}
# v.predictions_df['Cover_Type'].value_counts()
# 2    269711
# 1    219238
# 3     28927
# 7     19343
# 6     16676
# 5     10417
# 4      1580


#
# loguru init
#
class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


# when using loguru + pytest it's needed explicit propagation to the standard logger
# https://github.com/Delgan/loguru/issues/59#issuecomment-466532983
logger.add(PropagateHandler(), format="{message}")

# globals for debugging only(!) in qtconsole
# Warning: autoreload leads to them being reset dynamically after a code change
global g_settings
g_settings = {}


#
# Main entry
#
@click.command()
@click.option(
    "-d",
    "--dataset_path",
    default="data/only2krows",  # "data",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
)
@click.option(
    "-t",
    "--train_cfg",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("-a", "--autoreload", is_flag=True, default=False, help="Enable autoreload for qtconsole")
def run(**opts):
    """Main entry point."""
    if opts["autoreload"]:
        utils.autoreload()
        return
    global g_settings  # for debug
    g_settings = {
        item: getattr(settings_py, item)
        for item in dir(settings_py)
        if not item.startswith("__") and not item.endswith("__")
    }
    g_settings.update(opts)
    g_settings = utils.dotdict(g_settings)
    utils.process_settings(g_settings)
    g_settings.vars = utils.dotdict({})  # for debug
    # print("g_settings:", g_settings)
    # sys.modules['forest_cover_type']
    #
    g_settings.use_mlflow = use_mlflow and g_settings.use_mlflow
    if g_settings.use_mlflow:
        #
        # mlflow
        #
        # https://www.mlflow.org/docs/latest/tracking.html
        logger.info("mlflow is enabled")
        mlflow.set_experiment(g_settings.mode)
        # mlflow.sklearn.autolog(log_models=False, silent=True) # looks buggy and slow
        parent_run_name = "parent_run"
        if g_settings.feature_engineering:
            parent_run_name = g_settings.feature_engineering
        parent_run = mlflow.start_run(run_name=parent_run_name, description="", tags=opts)
        mlflow.log_param("parent", "yes")
        mlflow.log_param("version", __version__)
        mlflow.log_artifact("forest_cover_type/settings.py")
        client = MlflowClient()
        # https://www.mlflow.org/docs/latest/tracking.html#system-tags
        client.set_tag(run_id=parent_run.info.run_id, key="mlflow.user", value="")
        client.set_tag(run_id=parent_run.info.run_id, key="mlflow.source.git.commit", value=__version__)
        if g_settings.train_cfg is not None:
            mlflow.log_artifact(g_settings.train_cfg)
        mlflow.log_param("g_settings", g_settings)
    #
    # g_settings.mode = "xgb"
    logger.info(f"mode: {g_settings.mode}")
    #
    # mode
    #
    if g_settings.mode == "kfold":
        processed = preprocessing_v1.run(g_settings)
        for run_n in g_settings.runs.keys():
            if g_settings.use_mlflow:
                run_name = f"{run_n} {g_settings.runs[run_n].classifier}"
                if g_settings.feature_engineering:
                    run_name += " " + g_settings.feature_engineering
                nested_run = mlflow.start_run(run_name=run_name, nested=True)
                client.set_tag(run_id=nested_run.info.run_id, key="mlflow.user", value="")
                mlflow.log_param("nested_run", "yes")
            #
            # kfold
            #
            train_v1.kfold(g_settings, processed, run_n=run_n)
            if g_settings.use_mlflow:
                mlflow.end_run()
        # sys.exit()
    elif g_settings.mode == "xgb":
        #
        # xgb
        #
        # 0.76459
        g_settings.dataset_path = "data"
        g_settings.get_kaggle_score = True
        g_settings.create_submission_file = True
        g_settings.use_pl = False
        g_settings.feature_engineering = "fe_2"
        processed = preprocessing_v1.run(g_settings)
        train_v1.xgb_full(g_settings, processed)
    else:
        #
        # default: %run -m forest_cover_type
        #
        # print("g_settings:", g_settings)
        # all ExtraTreesClassifier > all RandomForestClassifier
        # all ExtraTreesClassifier (200) 0.83067
        # all ExtraTreesClassifier (50) 0.82762

        g_settings.use_booster = True  # 0.79->0.83
        g_settings.dataset_path = "data"
        g_settings.create_submission_file = True
        g_settings.load_test_csv = True
        g_settings.search_coef = False
        g_settings.use_pl = False
        g_settings.use_cyclic_pl = False
        if g_settings.use_cyclic_pl:
            g_settings.use_booster = True
        g_settings.save_new_labels = False
        g_settings.get_kaggle_score = True
        g_settings.feature_engineering = "fe_2"
        n_estim = 1000  # 200 -> 0.83124, 1000 -> 0.83169
        g_settings.clf_n_estimators = n_estim
        g_settings.booster_n_estimators_1 = n_estim
        g_settings.booster_n_estimators_2 = n_estim
        g_settings.max_depth = None
        g_settings.n_jobs = -1
        processed = preprocessing_v1.run(g_settings)
        g_settings.vars.processed = processed
        X_train, y = processed["train_dataframes"][0]
        if g_settings.use_pl:
            classifiers = train_v1.run_pl(g_settings, processed)
            g_settings.use_booster = False  # for predict_v1.run
            # g_settings.create_submission_file = False
        elif g_settings.use_cyclic_pl:
            assert len(processed["train_dataframes"]) == 3
            classifiers = train_v1.run(g_settings, processed["train_dataframes"])
            processed = predict_v1.with_new_labels(g_settings, classifiers, processed)
            classifiers = train_v1.run(g_settings, processed["train_dataframes"])
            processed = predict_v1.with_new_labels(g_settings, classifiers, processed)
            classifiers = train_v1.run(g_settings, processed["train_dataframes"])
            processed = predict_v1.with_new_labels(g_settings, classifiers, processed)
            classifiers = train_v1.run(g_settings, processed["train_dataframes"])
            processed = predict_v1.with_new_labels(g_settings, classifiers, processed)
            classifiers = train_v1.run(g_settings, processed["train_dataframes"])
            processed = predict_v1.with_new_labels(g_settings, classifiers, processed)
            X_test = processed["test_dataframe"]
            predictions_df = predict_v1.run(g_settings, classifiers, X_test, processed["sub_dataframe"])
            g_settings.vars.predictions_df = predictions_df
            kaggle_utils.create_sub_file(predictions_df, g_settings)
            kaggle_utils.upload_and_get_score(g_settings)
            g_settings.create_submission_file = False

        else:
            classifiers = train_v1.run(g_settings, processed["train_dataframes"])
            predictions_df = predict_v1.run(g_settings, classifiers, X_train)
            g_settings.vars.predictions_df = predictions_df
            acc_on_train = accuracy_score(y, predictions_df).round(5)
            logger.info(f"acc_on_train: {acc_on_train}")
            f1_w = f1_score(y, predictions_df, average="weighted")
            logger.info(f"f1_w: {f1_w}")

        if g_settings.use_mlflow:
            mlflow.log_metric("acc_on_train", acc_on_train) if g_settings.use_mlflow else None
        if g_settings.create_submission_file:
            X_test = processed["test_dataframe"]
            if g_settings.search_coef:
                for coef_1_diff in [-0.2, -0.1, 0.1, 0.2]:  # 1.4 > 1.2,
                    for coef_2_diff in [-0.2, -0.1, 0.1, 0.2]:  # 3.8 > 4
                        for coef_3_diff in [-0.2, -0.1, 0.1, 0.2]:  # 3.8 > 3.4
                            g_settings.coef_1 = 1.3 + coef_1_diff
                            g_settings.coef_2 = 3.9 + coef_2_diff
                            g_settings.coef_3 = 3.6 + coef_3_diff
                            logger.info(
                                f"coef_1: {g_settings.coef_1}, coef_2: {g_settings.coef_2}, coef_3: {g_settings.coef_3}"
                            )
                            predictions_df = predict_v1.run(
                                g_settings, classifiers, X_test, processed["sub_dataframe"]
                            )
                            kaggle_utils.create_sub_file(predictions_df, g_settings)
                            kaggle_utils.upload_and_get_score(g_settings)
                            logger.info("45 seconds sleep")
                            time.sleep(45)
            else:
                predictions_df = predict_v1.run(g_settings, classifiers, X_test, processed["sub_dataframe"])
                g_settings.vars.predictions_df = predictions_df
                if g_settings.save_new_labels:
                    newdf = g_settings.vars.df_test.merge(predictions_df, on="Id")
                    # newdf.shape (565892, 60)
                    new_big_train_df = g_settings.vars.df_train.append(newdf, ignore_index=True)
                    # new_big_train_df.shape (581012, 60)
                    logger.info(f"new_big_train_df.shape: {new_big_train_df.shape}")
                    new_train_fn = "new_big_train.csv"
                    new_big_train_df.to_csv(new_train_fn, index=False)
                    logger.info(f"Created {new_train_fn}.")
                else:
                    kaggle_utils.create_sub_file(predictions_df, g_settings)
                    if g_settings.get_kaggle_score:
                        kaggle_utils.upload_and_get_score(g_settings)
    if g_settings.use_mlflow:
        if g_settings.use_logfile:
            mlflow.log_artifact("file.log")
        mlflow.end_run()


if __name__ == "__main__":
    # When running from qtconsole with the wrong path command like:
    # %run -m forest_cover_type.runner -d asd
    # with just run() it shows list of big annoying exceptions.
    # So here is the temporary(?) solution to make it less annoying.
    try:
        run()
    except SystemExit as e:
        if len(str(e)) == 0:
            # print("no error")
            pass
        elif len(str(e)) == 1 and str(e) == "0":
            # Reached end of code
            pass
        else:
            # show minimum
            # write to log/history?
            traceback.print_exc(limit=1, chain=False)
            # logger.error(traceback.format_exc(limit=2, chain=False))
