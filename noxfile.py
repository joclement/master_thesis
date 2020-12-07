import tempfile

import nox
import nox_poetry.patch  # noqa F401

nox.options.sessions = "lint", "mypy", "tests"

LOCATIONS = "src", "tests", "noxfile.py"


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python="3.8")
def tests(session):
    args = session.posargs or ["-n", "3", "--cov", "-m", "not e2e"]
    session.install(".")
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-xdist")
    session.run("pytest", *args)


@nox.session(python="3.8")
def lint(session):
    session.install("flake8", "flake8-black", "flake8-import-order")
    session.run("flake8", *LOCATIONS)


@nox.session(python="3.8")
def black(session):
    session.install("black")
    session.run("black", *LOCATIONS)


@nox.session(python="3.8")
def mypy(session):
    session.install("mypy")
    session.run("mypy", *LOCATIONS)
