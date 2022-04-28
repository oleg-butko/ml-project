import sys
import argparse, os
from loguru import logger  # type:ignore

# fix to run debugger in vscode when current dir is not the same as in terminal
curdir = os.getcwd()
print("getcwd:", curdir)
add_dir = os.path.abspath(os.path.join(os.getcwd(), "010", "ml-project"))
sys.path.insert(0, os.path.abspath(add_dir))

# from forest_cover_type.package_one import module_one
from forest_cover_type.preprocessing import preprocessing_v1
from forest_cover_type.train import train_v1
from forest_cover_type.predict import predict_v1
from forest_cover_type.report import kaggle_utils

# globals for debugging only(!) in qtconsole
# these are being reset after code changes bc of autoreload
global processed, classifiers, X_test, predictions_df
processed = None
classifiers = None
X_test = None
predictions_df = None


def autoreload():
    """For jupyter qtconsole to autoreload the changed module"""
    get_ipython().run_line_magic("load_ext", "autoreload")  # type:ignore
    get_ipython().run_line_magic("autoreload", "2")  # type:ignore


def run():
    """Basic entry point."""
    global processed, classifiers, X_test, predictions_df
    print("preprocessing")
    processed = preprocessing_v1.run()
    print("train")
    classifiers = train_v1.run(processed["train_dataframes"])
    # sys.exit()
    assert len(classifiers.items()) == 3
    print("predict")
    X_test = processed["test_dataframe"]
    predictions_df = predict_v1.run(classifiers, X_test, processed["sub_dataframe"])
    kaggle_utils.create_sub_file(predictions_df)


def step1():
    print("step1")
    parser = argparse.ArgumentParser(description="Say hi.")
    parser.add_argument("target", type=str, help="the name of the target")
    parser.add_argument("--end", dest="end", default="!", help="sum the integers (default: find the max)")
    args = parser.parse_args()
    # some_function(args.target, end=args.end)


if __name__ == "__main__":
    run()
