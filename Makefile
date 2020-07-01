TESTDIR=tests
TESTS=$(TESTDIR)/residual_test.py $(TESTDIR)/mesh_test.py
PYTEST=pytest
PYTESTFLGS=-s

PIP=pip3
PIPFLGS=install -e .

.PHONY: all
all: compile test

.PHONY: compile
compile:
	$(PIP) $(PIPFLGS)

.PHONY: test
test:
	$(PYTEST) $(PYTESTFLGS) $(TESTS)
