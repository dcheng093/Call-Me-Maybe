.PHONY: install run debug lint lint-strict
GREEN := \033[0;32m
YELLOW := \033[1;33m
RESET := \033[0m

install:
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	@uv sync
	@uv pip install flake8 mypy pydantic numpy

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	@echo "$(YELLOW)Cleaning build files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf .pytest_cache data/output

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --strict
