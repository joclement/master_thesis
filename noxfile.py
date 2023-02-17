import nox
from nox_poetry import session

nox.options.sessions = "lint", "mypy", "tests"

LOCATIONS = "src", "tests", "noxfile.py"


@session(python="3.8")
def tests(session):
    args = session.posargs or [
        "tests",
        "-n",
        "3",
        "--cov",
        "-m",
        "not e2e and not expensive",
    ]
    session.install(".")
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-xdist")
    session.run("pytest", *args)


@session(python="3.8")
def lint(session):
    session.install("flake8", "flake8-black", "flake8-import-order")
    session.run(
        "flake8",
        "--application-import-names",
        "jc_thesis_code",
        "--application-package-names",
        "jc-thesis-code",
        *LOCATIONS
    )


@session(python="3.8")
def black(session):
    session.install("black")
    session.run("black", *LOCATIONS)


@session(python="3.8")
def mypy(session):
    session.install("mypy")
    session.run("mypy", *LOCATIONS)
