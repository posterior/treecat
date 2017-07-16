.PHONY: all format lint test FORCE

PY_FILES := *.py $(shell find treecat doc -name '*.py')

all: lint

format: FORCE
	yapf -i -p $(PY_FILES)
	isort -i $(PY_FILES)

lint: FORCE
	flake8 $(PY_FILES)

test: lint FORCE
	cd treecat ; py.test -vrw

clean:
	treecat.generate clean

FORCE:

