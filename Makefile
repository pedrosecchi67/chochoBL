TESTS=$(TESTDIR)/residual_test.py $(TESTDIR)/mesh_test.py
PYTEST=pytest
PYTESTFLGS=-s

PYTHON=python3
BUILDSCRIPT=setup.py build

.PHONY: all
all: compile test

.PHONY: compile
compile:
	$(PYTHON) $(BUILDSCRIPT)

.PHONY: test
test:
	$(PYTEST) $(PYTESTFLGS) $(TESTS)
