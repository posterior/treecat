.PHONY: all format lint test FORCE

PY_FILES := *.py $(shell find treetree -name '*.py')

all:

format:
	yapf -i $(PY_FILES)

lint:
	flake8 $(PY_FILES)

test: lint FORCE
	py.test


FORCE:

