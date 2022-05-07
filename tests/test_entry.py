from click.testing import CliRunner
import logging
import pytest
from forest_cover_type import __version__, runner, settings


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_version():
    assert __version__ == "0.1.0"


def test_basic_full_cycle(cli_runner: CliRunner, caplog) -> None:
    """Basic test for the main entry point. Fixed test data on input -> the known accuracy on output."""
    assert settings.SEED == 0
    result = cli_runner.invoke(runner.run, ["--dataset_path", "wrong_path"])
    assert result.exit_code == 2
    result = cli_runner.invoke(runner.run, ["--train_cfg", "wrong_path"])
    assert result.exit_code == 2
    # print(result.exit_code)
    # print(result.output)
    # print("caplog", caplog.text)
    result = cli_runner.invoke(
        runner.run,
        [
            "--dataset_path",
            "tests/data_for_test",
            "--train_cfg",
            "cfg/basic_full_cycle_test.ini",
        ],
    )
    caplog.set_level(logging.INFO)
    assert "df_train.shape: (2400, 56), df_test.shape: (100, 55)" in caplog.text
    assert "acc_on_train: 0.67542" in caplog.text
    assert result.exit_code == 0
