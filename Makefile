.PHONY: up down logs build test shell

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f bot

build:
	docker compose build

test:
	docker compose run --rm bot pytest tests/ -v

shell:
	docker compose run --rm bot bash
