.PHONY: install quickstart reproduce-results reproduce-inference reproduce-operational-evidence validate-public-contract evidence-pipeline test

install:
	conda env create -f environment.yml

quickstart:
	conda run -n cano-bigdata2026 python scripts/quickstart.py

reproduce-results:
	conda run -n cano-bigdata2026 python scripts/reproduce_results.py

reproduce-inference:
	conda run -n cano-bigdata2026 python scripts/reproduce_inference.py

reproduce-operational-evidence:
	conda run -n cano-bigdata2026 python scripts/reproduce_operational_evidence.py

validate-public-contract:
	conda run -n cano-bigdata2026 python scripts/validate_public_contract.py

evidence-pipeline:
	@echo "Run scripts/run_evidence_pipeline.py with three checkpoint paths and prepared data."

test:
	conda run -n cano-bigdata2026 python -m pytest -q
