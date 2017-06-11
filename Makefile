.PHONY: all format lint test FORCE

PY_FILES := *.py $(shell find treecat -name '*.py')

all: lint

format:
	yapf -i $(PY_FILES)

lint:
	flake8 $(PY_FILES)

test: lint FORCE
	py.test -v treecat


FORCE:

