# global settings
# other settings: cfg/*.ini

SEED = 0
# mlflow doesn't work from qtconsole
use_mlflow = True
submission_fn = "sub.csv"
# %s will be replaced with run_name
submission_fn_tmpl = "sub %s.csv"
create_submission_file = True
use_logfile = False
