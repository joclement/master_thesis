import nox
import nox_poetry

nox.options.sessions = "lint", "mypy", "tests"

LOCATIONS = "src", "tests", "noxfile.py"


@nox.session(python="3.8")
@nox_poetry.session
def tests(session):
    args = session.posargs or [
        "tests",
        "-n",
        "3",
        "--cov",
        "-m",
        "not e2e",
        "-m",
        "not expensive",
    ]
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
