PYTHON ?= .venv/bin/python

.PHONY: help verify figures refit paper toy-prepare toy-train study-figures study-verify study-prepare study-run
help:
	@echo 'After installing requirements/figures.txt:'
	@echo '  make verify       Check saved artifact hashes'
	@echo '  make figures      Rebuild toy plots from saved fits (CPU)'
	@echo '  make refit        Refit saved toy surfaces and plot (CPU)'
	@echo '  make paper        Reproduce pinned cached paper Figure 2 (network/git)'
	@echo 'After installing requirements/toy.txt:'
	@echo '  make toy-prepare  Stage a fresh 30-network run without training'
	@echo '  make toy-train    Train/continue the prepared full run (Apple MPS)'
	@echo 'Characterization study:'
	@echo '  make study-verify  Check saved study artifact hashes'
	@echo '  make study-figures Rebuild the six main study figures (CPU)'
	@echo '  make study-prepare Stage a full 120-network reproduction without training'
	@echo '  make study-run     Execute/continue that staged study (Apple MPS)'

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
study-verify:
	$(PYTHON) scripts/reproduce.py study-verify
study-figures:
	$(PYTHON) scripts/reproduce.py study-figures
study-prepare:
	$(PYTHON) scripts/reproduce.py study-run --prepare-only
study-run:
	$(PYTHON) scripts/reproduce.py study-run --resume
