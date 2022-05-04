import sys, os
import traceback
from pathlib import Path
import click
from loguru import logger  # type:ignore
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hide warnings from sklearn about deprecation that mlflow shows
warnings.filterwarnings("ignore", category=DeprecationWarning)
# mlflow is not available from qtconsole (for debugging): %run -m forest_cover_type.runner
use_mlflow = True
try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError:
    use_mlflow = False


from . import settings as settings_py
from .preprocessing import preprocessing_v1
from .train import train_v1
from .predict import predict_v1
from .report import kaggle_utils


def autoreload():
    """For jupyter qtconsole to auto-reload the just changed module.
    Example: %run -m forest_cover_type.runner -a
    """
    get_ipython().run_line_magic("load_ext", "autoreload")  # type:ignore
    get_ipython().run_line_magic("autoreload", "2")  # type:ignore


class dotdict(dict):
    """dot.notation access to dictionary attributes
    to write settings.dataset_path instead of settings["dataset_path"]
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


# globals for debugging only(!) in qtconsole
# Warning: autoreload leads to them being reset after code changes
global settings_obj, glob
settings_obj = None
glob = dotdict({})


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
        autoreload()
        return
    global settings_obj, glob
    settings_obj = {
        item: getattr(settings_py, item)
        for item in dir(settings_py)
        if not item.startswith("__") and not item.endswith("__")
    }
    settings_obj.update(opts)
    settings_obj = dotdict(settings_obj)
    if use_mlflow:
        mlflow.start_run()
        mlflow.sklearn.autolog(log_models=False, silent=True)
        mlflow.log_param("command_line_arguments", opts)
        mlflow.log_artifact("forest_cover_type/settings.py")
        if settings_obj.train_cfg is not None:
            mlflow.log_artifact(settings_obj.train_cfg)
        mlflow.log_param("settings_obj", settings_obj)
    click.echo("preprocessing")
    processed = preprocessing_v1.run(settings_obj)
    glob.X_train, glob.y = processed["train_dataframes"][0]
    click.echo("train")
    settings_obj.use_booster = False
    settings_obj.clf_n_estimators = 5
    classifiers = train_v1.run(settings_obj, processed["train_dataframes"])
    click.echo("predict")
    predictions_df = predict_v1.run(settings_obj, classifiers, glob.X_train)
    glob.predictions_df = predictions_df
    acc_on_train = accuracy_score(glob.y, predictions_df)
    click.echo(f"acc_on_train: {acc_on_train}")
    mlflow.log_metric("acc_on_train", acc_on_train) if use_mlflow else None
    # sys.exit()
    if settings_obj.create_submission_file:
        X_test = processed["test_dataframe"]
        predictions_df = predict_v1.run(settings_obj, classifiers, X_test, processed["sub_dataframe"])
        glob.predictions_df = predictions_df
        kaggle_utils.create_sub_file(predictions_df)
    mlflow.end_run() if use_mlflow else None


if __name__ == "__main__":
    # when running from qtconsole with the wrong path command like:
    # %run forest_cover_type/runner.py -d asd
    # with just run() it shows list of big annoying exceptions.
    # So here is the temporary(?) solution to make it less annoying.
    try:
        run()
    except SystemExit as e:
        if len(str(e)) == 0:
            # sys.exit()
            # print("no error")
            pass
        elif len(str(e)) == 1 and str(e) == "0":
            # end of code
            pass
        else:
            # show minimum
            # write to log/history?
            traceback.print_exc(limit=1, chain=False)
