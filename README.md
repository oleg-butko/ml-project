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
    `poetry run run -d data/only2krows`


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

1. Use the [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction). You will solve the task of forest cover type prediction and compete with other participants. **(necessary condition, 0 points for the whole homework if not done)**
2. Format your homework as a Python package. Use an [src layout](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure) or choose some other layout that seems reasonable to you, explain your choice in the README file. Don't use Jupyter Notebooks for this homework. Instead, write your code in .py files. **(necessary condition, 0 points for the whole homework if not done)**
4. Use [Poetry](https://python-poetry.org/) to manage your package and dependencies. **(6 points)**
    1. When installing dependencies, think if these dependencies will be used to run scripts from your package, or you'll need them only for development purposes (such as testing, formatting code, etc.). For development dependencies, use [the dev option of add command](https://python-poetry.org/docs/cli/#add). If you decided not to use Poetry, list your dependencies in requirements.txt and requirements-dev.txt files. **(4 points)**

## TODO

3. (Still less than 30 commits)
   Publish your code to Github. **(necessary condition, 0 points for the whole homework if not done)**
    1. Commits should be small and pushed while you're working on a project (not at the last moment, since storing unpublished code locally for a long time is not reliable: imagine something bad happens to your PC and you lost all your code). Your repository should have at least 30 commits if you do all non-optional parts of this homework. **(12 points)**

5. Create a data folder and place the dataset there. **(necessary condition, 0 points for the whole homework if not done. *Note for reviewers: data folder won't be seen on GitHub if added to gitignore, it's OK, check gitignore*)**
    1. Don't forget to add your data to gitignore. **(5 points)**
    2. (optional) Write a script that will generate you an EDA report, e.g. with [pandas profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/)
6. Write a script that trains a model and saves it to a file. Your script should be runnable from the terminal, receive some arguments such as the path to data, model configurations, etc. To create CLI, you can use argparse, click (as in the demo), hydra, or some other alternatives. **(10 points)**
    1. (optional) Register your script in pyproject.toml. This way you can run it without specifying a full path to a script file. **(2 points)**
7. Choose some metrics to validate your model (at least 3) and calculate them after training. Use K-fold cross-validation. **(10 points maximum: 2 per metric + 4 for K-fold. *Note for reviewers: K-fold CV may be overwritten by nested CV if the 9th task is implemented, check the history of commits in this case. If more than 3 metrics were chosen, only 3 are graded*)**
8. Conduct experiments with your model. Track each experiment into MLFlow. Make a screenshot of the results in the MLFlow UI and include it in README. You can see the screenshot example below, but in your case, it may be more complex than that. Choose the best configuration with respect to a single metric (most important of all metrics you calculate, according to your opinion). 
    1. Try at least three different sets of hyperparameters for each model. **(3 points)**
    2. Try at least two different feature engineering techniques for each model. **(4 points)**
    3. Try at least two different ML models. **(4 points)**
9. Instead of tuning hyperparameters manually, use automatic hyperparameter search for each model ...
10. In your README, write instructions on how to run your code (training script and optionally other scripts you created, such as EDA)...
11..15 (optional)


