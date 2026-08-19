.PHONY: install quickstart reproduce-results reproduce-inference evidence-pipeline test

install:
	conda env create -f environment.yml

quickstart:
	conda run -n cano-bigdata2026 python scripts/quickstart.py

reproduce-results:
	conda run -n cano-bigdata2026 python scripts/reproduce_results.py

reproduce-inference:
	conda run -n cano-bigdata2026 python scripts/reproduce_inference.py

evidence-pipeline:
	@echo "Run scripts/run_evidence_pipeline.py with three checkpoint paths and prepared data."

test:
	conda run -n cano-bigdata2026 pytest -q
