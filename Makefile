.PHONY: setup collect-2025 collect-2026 status snapshots test lint

setup:
	uv sync --extra dev --extra ml
	cp -n .env.example .env || true

collect-2025:
	uv run flowcast collect --start 2025-08-01 --end 2025-11-01
	uv run flowcast binance --start 2025-08-01 --end 2025-11-01

collect-2026:
	uv run flowcast collect --start 2026-07-01 --end 2026-10-01
	uv run flowcast binance --start 2026-07-01 --end 2026-10-01

status:
	uv run flowcast status

snapshots:
	uv run flowcast snapshots --pools SUI_USDC

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format src tests
