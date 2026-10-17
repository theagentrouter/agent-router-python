.PHONY: install test lint format surface llms models smoke build release local-gateway apps apps-live

install:  # theagentrouter-core (the SDK); add the local gateway with make local-gateway AIGW=...
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:
	.venv/bin/ruff format . && .venv/bin/ruff check --fix .

surface:
	.venv/bin/python scripts/surface.py --write

llms:  # llms.txt and llms-full.txt, from the README, docs, examples and skills
	.venv/bin/python scripts/build_llms.py

models:
	.venv/bin/python scripts/sync_models.py

apps:  # the apps under apps/ on real LiteLLM and on this SDK, offline: make apps [SDK=litellm|theagentrouter] [APP=...]
	scripts/apps.sh

apps-live:  # the apps against real providers on both SDKs; keys from the environment or .env: make apps-live [SDK=...]
	scripts/apps_live.sh

smoke:  # real providers; keys from the environment or a local .env
	@set -a; [ -f .env ] && . ./.env; set +a; .venv/bin/python scripts/smoke_real.py

build:
	.venv/bin/python -m build && .venv/bin/python -m build packages/theagentrouter --outdir dist

release:  # version, changelog date, checks, commit and tag; never pushes: make release V=1.2.3 [AMEND=1]
	@test -n "$(V)" || (echo "usage: make release V=1.2.3 [AMEND=1]" && exit 1)
	.venv/bin/python scripts/release.py $(V) $(if $(AMEND),--amend)

local-gateway:  # put an aigw program in the local-gateway package and install it for development: make local-gateway AIGW=/path/to/aigw
	@test -n "$(AIGW)" || (echo "usage: make local-gateway AIGW=/path/to/aigw" && exit 1)
	mkdir -p local-gateway/src/theagentrouter_local_gateway/bin
	cp "$(AIGW)" local-gateway/src/theagentrouter_local_gateway/bin/aigw
	chmod +x local-gateway/src/theagentrouter_local_gateway/bin/aigw
	.venv/bin/pip install -e local-gateway
