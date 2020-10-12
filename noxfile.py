import nox

nox.options.sessions = "lint", "mypy", "tests"

LOCATIONS = "src", "tests", "noxfile.py"


@nox.session(python=["3.8"])
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")


@nox.session(python=["3.8"])
def lint(session):
    args = session.posargs or LOCATIONS
    session.install("flake8", "flake8-black", "flake8-import-order")
    session.run("flake8", *args)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or LOCATIONS
    session.install("black")
    session.run("black", *args)


@nox.session(python="3.8")
def mypy(session):
    args = session.posargs or LOCATIONS
    session.install("mypy")
    session.run("mypy", *args)
