.PHONY: test lint smoke reproduce report verify paper

test:
	python -m pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check .

smoke:
	python train_and_eval.py --steps 2 --seeds 11 --eval-size 8 --latency-repeats 3 --output runs/smoke

reproduce:
	python train_and_eval.py --steps 300 --seeds 11 22 33 --checkpoints --output runs/reference

report:
	python scripts/summarize.py results/metrics.json
	python scripts/audit_intents.py

verify:
	python scripts/verify_results.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
