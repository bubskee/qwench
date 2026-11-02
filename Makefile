PYTHON ?= .venv/bin/python

.PHONY: help verify figures refit paper toy-prepare toy-train
help:
	@echo 'After installing requirements/figures.txt:'
	@echo '  make verify       Check saved artifact hashes'
	@echo '  make figures      Rebuild toy plots from saved fits (CPU)'
	@echo '  make refit        Refit saved toy surfaces and plot (CPU)'
	@echo '  make paper        Reproduce pinned cached paper Figure 2 (network/git)'
	@echo 'After installing requirements/toy.txt:'
	@echo '  make toy-prepare  Stage a fresh 30-network run without training'
	@echo '  make toy-train    Train/continue the prepared full run (Apple MPS)'

verify:
	$(PYTHON) scripts/reproduce.py verify
figures:
	$(PYTHON) scripts/reproduce.py toy-figures
refit:
	$(PYTHON) scripts/reproduce.py toy-refit
paper:
	$(PYTHON) scripts/reproduce.py paper
toy-prepare:
	$(PYTHON) scripts/reproduce.py toy-train --prepare-only
toy-train:
	$(PYTHON) scripts/reproduce.py toy-train --resume
