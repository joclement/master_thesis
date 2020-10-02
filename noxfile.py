import nox

nox.options.sessions = "lint", "tests"

LOCATIONS = "src", "tests", "noxfile.py"


@nox.session(python=["3.8"])
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")


@nox.session(python=["3.8"])
def lint(session):
    args = session.posargs or LOCATIONS
    session.install("flake8")
    session.run("flake8", *args)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or LOCATIONS
    session.install("black")
    session.run("black", *args)
