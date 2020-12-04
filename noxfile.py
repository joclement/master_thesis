import tempfile

import nox

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


@nox.session(python=["3.8"])
def tests(session):
    args = session.posargs or ["-n", "3", "--cov", "-m", "not e2e"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session, "coverage[toml]", "pytest", "pytest-cov", "pytest-xdist"
    )
    session.run("pytest", *args)


@nox.session(python=["3.8"])
def lint(session):
    install_with_constraints(session, "flake8", "flake8-black", "flake8-import-order")
    session.run("flake8", *LOCATIONS)


@nox.session(python="3.8")
def black(session):
    install_with_constraints(session, "black")
    session.run("black", *LOCATIONS)


@nox.session(python="3.8")
def mypy(session):
    install_with_constraints(session, "mypy")
    session.run("mypy", *LOCATIONS)
