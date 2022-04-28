import numpy as np

filename = "text.txt"
SEED = 0
submission_fn = "sub.csv"
PATH = "../ml-project/data/"
PATH_2 = "./010/ml-project/data/"

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=115,
    threshold=1000,
    formatter=dict(float_kind=lambda x: "%6.3f" % x),
)

