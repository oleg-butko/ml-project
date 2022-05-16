## Python Project Template

I tried a couple templates and chose to go with [this one](https://github.com/MislavJaksic/Python-Project-Template) because it was pretty simple.


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
    2. `%run -m forest_cover_type.runner -d data/only2krows -t cfg/kfold_fast.ini.txt`
    3. `%run -m forest_cover_type.runner -a` -- to enable auto-update of changed modules
    4. poetry use:
    5. `poetry run entry -d data/only2krows`
5. When adding mlflow on Windows OS there was the error with pywin32 package. The solution I found was to add pywin32 first with `poetry add pywin32 --optional` and then `poetry add mlflow`. Remove pywin32 if you're not on Windows and have troubles with pywin32.
6. MLflow doesn't work when running from qtconsole. But it's not needed there anyway. The `settings.py` has option to disable a few things including mlflow.
7. The most common command: `poetry run entry -d data -t cfg/kfold.ini.txt`
8.  The `data` folder must have: train.csv (~2MB) test.csv (~72MB)
9.  The subfolder `data/only2krows` could have a smaller subset of the full data with only 2-3k rows. In this case the command will be: `poetry run entry -d data/only2krows -t cfg/kfold.ini.txt`
10. `poetry run pytest` doesn't work. It wasn't updated for the recent changes.
11. `nox` and `mypy` probably doesn't work too.

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
   1. `poetry run entry -d data -t cfg/using_cache.ini`
   2. `poetry run entry -d data -t cfg/kfold.ini.txt`
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
11-14 Not done


## About score 0.83169

1. Я взял [этот ноутбук](https://www.kaggle.com/code/shouldnotbehere/two-models-random-forests), который дает 0.82532 и пытался понять как у него это получилось и можно ли его как-то улучшить.
2. Идея там достаточно простая. Тестовая выборка сильно смещена в сторону двух классов - 1 и 2.
   В трейне (где поставлены 15к меток) все классы представленны равномерно по 2160 штук каждый.
   В тесте (где метки неизвестны) после любой простой модели будет видно большое смещение в сторону 1 и 2:
 ```
    2    269711
    1    219238
    3     28927
    7     19343
    6     16676
    5     10417
    4      1580
```
   1. Поэтому идея в том, что надо уметь хорошо различать эти два класса. Делается 3 модели:
      1. Для всех меток .
      2. Для 1 и 2.
      3. Для остальных 3 4 5 6 7 (эта модель мало улучшает, но хоть что-то).
   2. Результаты predict_proba смешиваются через некие коэффициенты, которые подбираются экспериментально.
3. Я сделал автоматический подбор коэффициентов, т.е. скрипт сам меняет их, загружает сабмит на каггл и показывает что получилось. Это дало максимальный эффект. Все остальные попытки ни дали прибавления или были хуже.
4. Я попробовал xgboost. Увидел, что он не дает прибавления по сравнению с ExtraTreesClassifier или (скорее всего) я просто не смог с ним разобраться в короткое время. Первая простая xgboost модель, которая на все классы, выдавала результат хуже, чем ExtraTreesClassifier и намного дольше считала. Так что я его оставил на потом.
5. Потом я вспомнил, что есть волшебная техника называемая `pseudo labeling`. Я попытался воспроизвести этот метод, который заключается в том, что новые предсказанные метки классов добавляем в трейн и заново обучаемся на новом увеличенном трейне. Для этого соревнования этот метод выглядел очень подходящим, т.к. тест в десятки раз больше чем трейн. Возможно я неправильно реализовал этот метод, но улучшения он не дал, хотя и ухудшения не было.

    


