.PHONY: install test lint fmt build up down logs shell clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

build:
	docker compose build

up:
	docker compose up

up-d:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

shell:
	docker compose run --rm --entrypoint /bin/sh amms

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
