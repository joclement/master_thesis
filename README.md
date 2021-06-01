# Implementation for Master Thesis

[![Tests](https://github.com/flyingdutchman23/thesis_implementation/workflows/Tests/badge.svg)](https://github.com/flyingdutchman23/thesis_implementation/actions?workflow=Tests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the code to the master's thesis from Joris Clement with the title "Evaluation
of Classification Algorithms for Partial Discharge Diagnosis in Gas-Insulated HVDC
Systems".
The code is written in Python. Poetry is used for the management of the dependencies and
to execute the scripts.
Check the [Poetry documentation](https://python-poetry.org/) on how to install it.
When Poetry is installed execute the following:
```bash
poetry install
```

With nox the tests and other checks can be run. Therefore nox needs to be installed.
See the documentation [here](https://nox.thea.codes/en/stable/).
Then just type `nox` on the command line to run the tests and checks.


After the dependencies are installed and the scripts can be executed. Run for e.g.
```bash
poetry run classify config/fingerprints
```
to train and validate some of the feature-based models.
The data is needed to do that. The data is contained in the submodule `data` and not in
this repo directly due to its disk usage of more than 5GB. It was further required to
keep the data private.
Example data can be found `./testdata/`.
