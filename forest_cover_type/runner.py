import sys, os, traceback, warnings, logging
from pathlib import Path
import click
from loguru import logger  # type:ignore
from sklearn.metrics import accuracy_score


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

# Example of correct call from qtconsole:
# %run -m forest_cover_type.runner -a
# %run -m forest_cover_type.runner -d data/only2krows -t cfg/kfold.ini


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
    g_settings.vars = utils.dotdict({})
    utils.process_settings(g_settings)
    # print("g_settings:", g_settings)
    # print("sys.modules[__name__]:", sys.modules[__name__])
    # sys.modules['forest_cover_type'].runner
    # sys.exit()

    #
    # kaggle_utils.upload_and_get_score(g_settings)
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
    else:
        #
        # simple default: %run -m forest_cover_type
        #
        # print("g_settings:", g_settings)
        g_settings.dataset_path = "data"
        g_settings.create_submission_file = True
        g_settings.get_kaggle_score = True
        g_settings.feature_engineering = "fe_2"
        g_settings.clf_n_estimators = 100
        g_settings.max_depth = None
        g_settings.use_booster = True
        g_settings.n_jobs = -1
        processed = preprocessing_v1.run(g_settings)
        g_settings.vars.X_train, g_settings.vars.y = processed["train_dataframes"][0]
        classifiers = train_v1.run(g_settings, processed["train_dataframes"])
        X_train_df = processed["train_dataframes"][0][0]
        predictions_df = predict_v1.run(g_settings, classifiers, X_train_df)
        g_settings.vars.predictions_df = predictions_df
        acc_on_train = accuracy_score(g_settings.vars.y, predictions_df).round(5)
        logger.info(f"acc_on_train: {acc_on_train}")
        if g_settings.use_mlflow:
            mlflow.log_metric("acc_on_train", acc_on_train) if g_settings.use_mlflow else None
        if g_settings.create_submission_file:
            X_test = processed["test_dataframe"]
            predictions_df = predict_v1.run(g_settings, classifiers, X_test, processed["sub_dataframe"])
            g_settings.vars.predictions_df = predictions_df
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
