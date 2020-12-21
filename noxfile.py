import nox
import nox_poetry.patch  # noqa F401

nox.options.sessions = "lint", "mypy"

LOCATIONS = "src", "tests", "noxfile.py"


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
