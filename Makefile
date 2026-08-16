PYTHON ?= python3
CONFIG ?= config/project.yaml
CONSTRAINTS ?= constraints-tested.txt

.PHONY: install audit features eda train validate run test clean

install:
	$(PYTHON) -m pip install -c $(CONSTRAINTS) -e ".[dev,kaggle]"

audit:
	$(PYTHON) -m credit_scorecard audit-data --config $(CONFIG)

features:
	$(PYTHON) -m credit_scorecard build-features --config $(CONFIG)

eda:
	$(PYTHON) -m credit_scorecard eda --config $(CONFIG)

train:
	$(PYTHON) -m credit_scorecard train --config $(CONFIG)

validate:
	$(PYTHON) -m credit_scorecard validate --config $(CONFIG)

run:
	$(PYTHON) -m credit_scorecard run-all --config $(CONFIG)

test:
	$(PYTHON) -m pytest

clean:
	$(PYTHON) -m credit_scorecard clean-generated --config $(CONFIG)
