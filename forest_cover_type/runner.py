import sys
from loguru import logger  # type:ignore
from forest_cover_type.package_one import module_one
import argparse, os
import pandas as pd


def load_data():
    print("load_data", os.getcwd())
    print("pd.__version__", pd.__version__)
    PATH = "./data/"
    df = pd.read_csv(PATH + "train.csv")
    print(df.head(1))


def step1():
    print("step1")
    parser = argparse.ArgumentParser(description="Say hi.")
    parser.add_argument("target", type=str, help="the name of the target")
    parser.add_argument("--end", dest="end", default="!", help="sum the integers (default: find the max)")
    args = parser.parse_args()
    # some_function(args.target, end=args.end)


def main(args):
    """main() will be run if you run this script directly"""
    x = 2
    y = 7
    print("main")
    print(module_one.add(x, y))  # -> 9
    print(module_one.multiply(x, y))  # -> 14


def run():
    """Entry point for the runnable script."""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    """main calls run()."""
    run()
