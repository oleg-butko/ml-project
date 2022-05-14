import sys, time, subprocess, io
import pandas as pd
from loguru import logger  # type:ignore
from .. import utils

glob = utils.dotdict({})
# global glob


def create_sub_file(df, settings):
    df.to_csv(settings.submission_fn, index=False)


def upload_and_get_score(settings):
    """Get publicScore for the file settings.submission_fn"""
    sub_msg = time.strftime("%Y-%m-%d-%H-%M-%S")
    cmd_1 = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "forest-cover-type-prediction",
        "-f",
        settings.submission_fn,
        "-m",
        sub_msg,
    ]
    logger.info(f"Uploading: {settings.submission_fn}")
    res_1 = subprocess.check_output(cmd_1, shell=True).decode(sys.stdout.encoding).strip()
    assert res_1 == "Successfully submitted to Forest Cover Type Prediction"
    logger.info(f"Pause for a few seconds.")
    time.sleep(3)
    logger.info(f"Getting new score.")
    cmd_2 = [
        "kaggle",
        "competitions",
        "submissions",
        "--csv",
        "-c",
        "forest-cover-type-prediction",
    ]
    res_2 = subprocess.check_output(cmd_2, shell=True).decode(sys.stdout.encoding)
    df = pd.read_csv(io.StringIO(res_2))
    score = df[df.description == sub_msg].publicScore.values[0]
    logger.info(f"Score: {score}")
    return score
