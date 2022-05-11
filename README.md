## Python Project Template

I have tried a couple templates and chose to go with this one because it was pretty simple.
[Python Project Template](https://github.com/MislavJaksic/Python-Project-Template)

## Notes

1. I took this [very simple notebook](https://www.kaggle.com/code/shouldnotbehere/two-models-random-forests) 
as a base to make a simple working structure with 4 steps: preprocessing, train, predict, report.
2. Steps how to run docs/make.bat from vscode:
    1. Make sure you opened the poetry virtual env (poetry env info)
    2. In terminal change the current dir to docs
    3. Run make.bat html, the result should be in docs/build/html
3. How to use qtconsole:
    1. poetry shell
    2. In the shell: jupyter-qtconsole.exe --style monokai --no-confirm-exit
    3. In the qtconsole: %run forest_cover_type/runner.py
4. Using CLI:
    1. qtconsole use:
     `%run forest_cover_type/runner.py -d data/only2krows`

     `%run forest_cover_type/runner.py -a`
    2. poetry use:
    `poetry run entry -d data/only2krows`
5. When adding mlflow on Windows OS there was the error with pywin32 package. The solution I found was to add pywin32 first with `poetry add pywin32 --optional` and then `poetry add mlflow`.
6. The command `%run forest_cover_type/runner.py` for qtconsole doesn't work now, because of using relative import "ImportError: attempted relative import with no known parent package". The command `%run -m forest_cover_type.runner -d data` works.
7. MLflow doesn't work when running from qtconsole. But it's not needed there anyway.


### Some basic commands:

```
poetry install

tests
poetry run pytest --durations=0
poetry run pytest --cov=forest_cover_type --cov-report=html tests

formatter
poetry run black .

build
poetry build

profiler
poetry shell
in shell: python -m cProfile forest_cover_type/runner.py
```



## Done

1. Use the [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction). 
2. Format your homework as a Python package
3. Publish your code to Github.
    1. Commits should be small and pushed while you're working on a project... at least 30 commits **(12 points)**
4. Use [Poetry](https://python-poetry.org/) to manage your package and dependencies. **(6 points)**
    1. use [the dev option of add command](https://python-poetry.org/docs/cli/#add). **(4 points)**
5. Create a data folder and place the dataset there.
    1. add your data to gitignore. **(5 points)**
6. Write a script that trains a model and saves it to a file. Your script should be runnable from the terminal, receive some arguments such as the path to data, model configurations, etc. To create CLI, you can use argparse, click (as in the demo), hydra, or some other alternatives. **(10 points)**
   1. poetry run entry -d data -t cfg/using_cache.ini
   2. poetry run entry -d data -t cfg/kfold.ini.txt
   3. (optional) Register your script in pyproject.toml **(2 points)**
7. Choose some metrics to validate your model (at least 3) and calculate them after training. Use K-fold cross-validation. **10 points maximum: 2 per metric + 4 for K-fold**
   1. ml-project/forest_cover_type/train/train_v1.py -> kfold
8. Conduct experiments with your model. Track each experiment into MLFlow. Make a screenshot of the results in the MLFlow UI and include it in README.
    1. Try at least three different sets of hyperparameters for each model. **(3 points)**
    2. Try at least two different feature engineering techniques for each model. **(4 points)**
    3. Try at least two different ML models. **(4 points)**
    4. Here is the screenshot https://github.com/oleg-butko/ml-project/blob/main/scrshot/mlflow.jpg
    6. kfold.ini.txt has sections named [run_1]..[run_n]. Each section has classifier parameter (2 different classifiers/models). The option for feature engineering is common for all sections. In the screenshot it's 2 runs (CLI calls) with different configs (feature engineering named fe_1 or fe_2).
9. Not done
10. In your README, write instructions on how to run your code (training script and optionally other scripts you created, such as EDA)


