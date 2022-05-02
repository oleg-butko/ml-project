import sys, os
import traceback
from pathlib import Path
import click
from loguru import logger  # type:ignore

# fix to run debugger in vscode when current dir is not the same as in terminal
# curdir = os.getcwd()
# print("getcwd:", curdir)
# add_dir = os.path.abspath(os.path.join(os.getcwd(), "010", "ml-project"))
# sys.path.insert(0, os.path.abspath(add_dir))

from forest_cover_type import settings
from forest_cover_type.preprocessing import preprocessing_v1
from forest_cover_type.train import train_v1
from forest_cover_type.predict import predict_v1
from forest_cover_type.report import kaggle_utils

# globals for debugging only(!) in qtconsole
# these are being reset after code changes bc of autoreload
global settings_obj, processed, classifiers, X_test, predictions_df
settings_obj = None
processed = None
classifiers = None
X_test = None
predictions_df = None


def autoreload():
    """For jupyter qtconsole to autoreload the changed module"""
    get_ipython().run_line_magic("load_ext", "autoreload")  # type:ignore
    get_ipython().run_line_magic("autoreload", "2")  # type:ignore


class dotdict(dict):
    """dot.notation access to dictionary attributes
    to write settings.dataset_path instead of settings["dataset_path"]"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


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
    """Basic entry point."""
    global settings_obj, processed, classifiers, X_test, predictions_df
    # print(opts)
    settings_obj = {
        item: getattr(settings, item)
        for item in dir(settings)
        if not item.startswith("__") and not item.endswith("__")
    }
    settings_obj.update(opts)
    # print("runner.py: settings_obj", settings_obj)
    settings_obj = dotdict(settings_obj)
    if opts["autoreload"]:
        autoreload()
        return
    print("preprocessing")
    processed = preprocessing_v1.run(settings_obj)
    print("train")
    classifiers = train_v1.run(settings_obj, processed["train_dataframes"])
    # sys.exit()
    print("predict")
    X_test = processed["test_dataframe"]
    predictions_df = predict_v1.run(settings_obj, classifiers, X_test, processed["sub_dataframe"])
    kaggle_utils.create_sub_file(predictions_df)


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
