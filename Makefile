TESTDIR=tests
TESTS=$(TESTDIR)/residual_test.py $(TESTDIR)/mesh_test.py $(TESTDIR)/findiff_test.py # $(TESTDIR)/newton_krylov.py
PYTEST=py.test
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
