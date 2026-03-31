.PHONY: install test benchmark lint format build clean

install: ## Install package in development mode alongside dev dependencies
	@pip install -e ".[dev]"

test: ## Execute the full Pytest QA regression suite
	@pytest tests/

benchmark: ## Run the generic GPU throughput benchmark tests
	@python scripts/benchmark.py --model microsoft/phi-2 --budget 0.3

lint: ## Execute static code analysis with ruff
	@ruff check nesion/ tests/

format: ## Automatically format all code bases with black formatting
	@black nesion/ tests/ scripts/

build: ## Compile pure python source codes into sdist and wheel formats 
	@python -m build

clean: ## Remove temporary python artifacts and compiled caches
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf dist/ build/ results/
