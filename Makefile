SHELL := /bin/sh

UV ?= uv
COMPOSE ?= docker compose
ENV_FILE ?= .env
DATA_DIR ?= data
COMPOSE_WITH_ENV = env -u NEO4J_USERNAME -u NEO4J_PASSWORD $(COMPOSE) --env-file "$(ENV_FILE)"

.PHONY: help install neo4j neo4j-reset run stop test clean

help:
	@printf '%s\n' \
		'make install  Install the project and test dependencies' \
		'make neo4j    Start Neo4j and wait until it is healthy' \
		'make neo4j-reset  Delete local Neo4j data and initialize it from ENV_FILE' \
		'make run      Start Neo4j, wait for health, then run the experiment' \
		'make stop     Stop the Neo4j container' \
		'make test     Run the offline test suite'

install:
	$(UV) sync --extra dev

neo4j:
	$(COMPOSE_WITH_ENV) up -d --wait neo4j

neo4j-reset:
	$(COMPOSE_WITH_ENV) down --volumes
	$(COMPOSE_WITH_ENV) up -d --wait neo4j

run: neo4j
	$(UV) run d-bridge --data-dir "$(DATA_DIR)" --env-file "$(ENV_FILE)"

stop:
	$(COMPOSE_WITH_ENV) stop neo4j

test:
	$(UV) run pytest -q

clean:
	$(COMPOSE_WITH_ENV) down
