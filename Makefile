COMPOSE := $(shell command -v podman >/dev/null 2>&1 && echo "podman compose" || echo "docker compose")

IMAGE := ghcr.io/chan/happytorch

.PHONY: run stop clean jupyter push

run:
	$(COMPOSE) up --build -d
	@echo ""
	@echo "HappyTorch is running!"
	@echo "   Open http://localhost:8000"
	@echo ""

jupyter:
	MODE=jupyter PORT=8888 $(COMPOSE) -f docker-compose.yml -f docker-compose.jupyter.yml up --build -d
	@echo ""
	@echo "HappyTorch JupyterLab is running!"
	@echo "   Open http://localhost:8888"
	@echo ""

stop:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	rm -f data/progress.json

push:
	docker build -t $(IMAGE):latest .
	docker push $(IMAGE):latest
