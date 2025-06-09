.PHONY: all pipe venv qvenv launch drone anim clean cache help

help:
	@echo "Available commands:"
	@echo "  make all         - Calls make pipeline"
	@echo "  make pipe        - Clean, setup venv, run, and clean"
	@echo "----- Virtual env -----"
	@echo "  make venv        - Create a venv and install requirements"
	@echo "  make qvenv       - Remove venv"
	@echo "----- Launch -----"
	@echo "  make launch      - Launch the project"
	@echo "  make drone       - Launch the drone fleet calculation script"
	@echo "  make anim        - Launch the animation script"
	@echo "----- Clean -----"
	@echo "  make clean-cache - Empties all caches"
	@echo "  make clean-all   - Remove all generated files and cache"
	@echo "----- Help -----"
	@echo "  make help        - Show this help message"

venv:
	python3 -m venv venv && \
	venv/bin/pip install --upgrade pip && \
	venv/bin/pip install -r requirements.txt

qvenv:
	rm -rf venv

launch:
	venv/bin/python launch.py

drone:
	venv/bin/python scripts/drone.py

anim:
	venv/bin/python scripts/animate.py

clean-cache:
	rm -rf cache/*	
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-all: clean-cache
	rm -rf assets/graph/geojson/*
	rm -rf assets/graph/graphml/*
	rm -rf assets/graph/img/*
	rm -rf assets/drone_fleet/*
	

pipe: clean-all venv launch qvenv

all: pipe
