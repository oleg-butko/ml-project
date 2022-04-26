## Python Project Template

[Python Project Template](https://github.com/MislavJaksic/Python-Project-Template)


```
poetry install

run
poetry run python ./forest_cover_type/runner.py
or just
poetry run forest_cover_type

tests
poetry run pytest --durations=0
poetry run pytest --cov=forest_cover_type --cov-report=html tests

linter
poetry run black .

build
poetry build

profiler
poetry shell
in shell: python -m cProfile forest_cover_type/runner.py


```
