"""Nox sessions."""

import tempfile
from typing import Any

import nox
from nox.sessions import Session


# nox.options.sessions = "black", "mypy"
nox.options.sessions = ["tests"]
locations = "forest_cover_type", "noxfile.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """Install packages constrained by Poetry's lock file.
    By default newest versions of packages are installed,
    but we use versions from poetry.lock instead to guarantee reproducibility of sessions.
    """
    with tempfile.NamedTemporaryFile(delete=False) as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python="3.10")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.10")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python="3.10")
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs
    # Here "--no-dev" is okay and works when using just "nox"
    # session.run("poetry", "install", "--no-dev", external=True)
    # But this command doesn't work with "--no-dev"
    # nox --reuse-existing-virtualenvs --no-install
    session.run("poetry", "install", external=True)
    install_with_constraints(session, "pytest")
    session.run("pytest", *args)
