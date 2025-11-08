# Fabric DT — Makefile
# Quickstart:
#   make install           # create venv + install deps
#   make run-api           # start DT API (http://127.0.0.1:8080)
#   make run-ui            # start Dashboard (http://127.0.0.1:8090)
#   make gen-nodes         # synthesize 100 node YAMLs into nodes/
#   make validate-nodes    # schema-validate nodes/
#   make plan              # plan jobs/jobs_10.yaml (dry-run)
#   make demo              # fire random demo jobs (local)
#   make montecarlo        # run Monte Carlo simulator
#   make chaos             # apply chaos events from sim/topology.yaml
#   make docker-launch     # (optional) approximate container emulation
#   make docker-clean      # stop/remove launched containers

# ---------- config ----------
PY     ?= python3
PIP    ?= pip3
VENV   ?= .venv
ACT    ?= . $(VENV)/bin/activate

API_HOST ?= 127.0.0.1
API_PORT ?= 8080
UI_HOST  ?= 127.0.0.1
UI_PORT  ?= 8090

NODES_DIR ?= nodes
JOBS_FILE ?= jobs/jobs_10.yaml
TOPO_FILE ?= sim/topology.yaml

# ---------- meta ----------
.PHONY: help install venv deps freeze clean format lint \
	run-api run-ui gen-nodes validate-nodes summarize-nodes export-csv \
	plan demo montecarlo chaos \
	docker-launch docker-clean

help:
	@echo "Targets:"
	@echo "  install          - create venv and install requirements"
	@echo "  run-api          - start DT API at http://$(API_HOST):$(API_PORT)"
	@echo "  run-ui           - start Dashboard at http://$(UI_HOST):$(UI_PORT)"
	@echo "  gen-nodes        - synthesize 100 realistic nodes into $(NODES_DIR)/"
	@echo "  validate-nodes   - validate nodes/*.yaml against schema"
	@echo "  summarize-nodes  - print inventory table"
	@echo "  export-csv       - export last plan(s) to CSV"
	@echo "  plan             - plan $(JOBS_FILE) locally (dry-run)"
	@echo "  demo             - send demo jobs (local); see vars NUM, WORKERS"
	@echo "  montecarlo       - run Monte Carlo simulation"
	@echo "  chaos            - apply chaos schedule from $(TOPO_FILE) (SCENARIO=name)"
	@echo "  docker-launch    - launch approx containers for nodes/"
	@echo "  docker-clean     - stop & remove launched containers"
	@echo "  format           - black/isort format"
	@echo "  lint             - ruff lint"
	@echo "  clean            - remove caches & build artifacts"

# ---------- env/deps ----------
venv:
	$(PY) -m venv $(VENV)

deps: requirements.txt | venv
	$(ACT); $(PIP) install -U pip
	$(ACT); $(PIP) install -r requirements.txt

install: deps
	@echo "✔ Environment ready."

freeze:
	$(ACT); $(PIP) freeze > requirements.lock.txt

# ---------- run DT ----------
run-api:
	$(ACT); FABRIC_API_HOST=$(API_HOST) FABRIC_API_PORT=$(API_PORT) $(PY) -m dt.api

run-ui:
	$(ACT); \
	if [ -z "$${FABRIC_DT_REMOTE+x}" ]; then \
		REMOTE="http://$(API_HOST):$(API_PORT)"; \
	else \
		REMOTE="$${FABRIC_DT_REMOTE}"; \
	fi; \
	FABRIC_UI_HOST=$(UI_HOST) FABRIC_UI_PORT=$(UI_PORT) FABRIC_DT_REMOTE="$$REMOTE" $(PY) -m ui.dashboard

# ---------- data generation / validation ----------
gen-nodes:
	$(ACT); $(PY) -m sim.gen_nodes --out-dir $(NODES_DIR) --count 100

validate-nodes:
	$(ACT); $(PY) -m tools.validate_nodes

summarize-nodes:
	$(ACT); $(PY) -m tools.summarize_nodes

export-csv:
	$(ACT); $(PY) -m tools.export_csv --in plans/last.json --out plots/last.csv

# ---------- planning ----------
plan:
	$(ACT); $(PY) -m planner.run_plan --job $(JOBS_FILE) --dry-run

# Demo knobs: NUM=50 WORKERS=4 QPS=2.0 DRY=0 REMOTE=http://127.0.0.1:8080
demo:
	$(ACT); $(PY) -m planner.submit_demo \
		-n $${NUM:-10} -w $${WORKERS:-1} --qps $${QPS:-0.0} \
		$$( [ "$${DRY:-1}" = "1" ] && echo "" || echo "--no-dry-run") \
		$$( [ -z "$${REMOTE}" ] && echo "" || echo "--remote $${REMOTE}" ) \
		--out-json plans/demo.json --out-csv plans/demo.csv

# ---------- simulation ----------
montecarlo:
	$(ACT); $(PY) -m sim.montecarlo --jobs $(JOBS_FILE) --trials $${TRIALS:-200} --out plans/montecarlo.json

chaos:
	$(ACT); $(PY) -m sim.chaos --topology $(TOPO_FILE) $$( [ -z "$$SCENARIO" ] && echo "" || echo "--scenario $$SCENARIO" ) --run

# ---------- docker (optional) ----------
# Requires: pip install docker, docker daemon running
docker-launch:
	$(ACT); $(PY) -m fabric_docker.launch_fabric --nodes $(NODES_DIR) --topology $(TOPO_FILE) --network fabric-net --image alpine:3.20 --prefix fab- --tc none

docker-clean:
	@echo "Stopping & removing containers with label/prefix 'fab-' on network fabric-net…"
	- docker ps -a --format '{{.Names}}' | grep '^fab-' | xargs -r docker rm -f
	- docker network rm fabric-net 2>/dev/null || true
	@echo "✔ Docker clean done."

# ---------- dev hygiene ----------
format:
	$(ACT); black dt planner sim tools ui fabric_docker || true
	$(ACT); isort dt planner sim tools ui fabric_docker || true

lint:
	$(ACT); ruff check dt planner sim tools ui fabric_docker || true

clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '*.pyc' -delete
	@rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	@echo "✔ Cleaned."

