.PHONY: install test coverage lint format clean demo release

install:
	pip install -e ".[dev,test]"

test:
	pytest

coverage:
	pytest --cov=episodic_memory --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/episodic_memory/

demo:
	python examples/demo.py

clean:
	rm -rf build/ dist/ *.egg-info .coverage .pytest_cache __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

release: test lint typecheck
	python -m build
	twine check dist/*
	echo "Ready to upload: python -m twine upload dist/*"

dev:
	pip install -e .