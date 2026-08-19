.PHONY: install quickstart reproduce-results test

install:
	conda env create -f environment.yml

quickstart:
	conda run -n cano-bigdata2026 python scripts/quickstart.py

reproduce-results:
	conda run -n cano-bigdata2026 python scripts/reproduce_results.py

test:
	conda run -n cano-bigdata2026 pytest -q
