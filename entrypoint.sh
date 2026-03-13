#!/bin/bash
set -e

MODE="${MODE:-web}"
mkdir -p /app/data

if [ "$MODE" = "jupyter" ]; then
    NOTEBOOKS_DIR=/app/notebooks
    TEMPLATES_DIR=/app/templates
    SOLUTIONS_DIR=/app/solutions

    mkdir -p "$NOTEBOOKS_DIR"

    echo "Resetting blank notebooks..."
    for f in "$TEMPLATES_DIR"/*.ipynb; do
        [ -f "$f" ] || continue
        cp -f "$f" "$NOTEBOOKS_DIR/$(basename "$f")"
    done

    echo "Syncing solution notebooks..."
    for f in "$SOLUTIONS_DIR"/*.ipynb; do
        [ -f "$f" ] || continue
        cp -f "$f" "$NOTEBOOKS_DIR/$(basename "$f")"
    done

    echo "Launching JupyterLab on port ${PORT:-8888}..."
    exec jupyter lab \
        --ip=0.0.0.0 \
        --port="${PORT:-8888}" \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password='' \
        --ServerApp.tornado_settings='{"headers": {"Content-Security-Policy": "frame-ancestors *"}}' \
        --ServerApp.allow_origin='*' \
        --ServerApp.disable_check_xsrf=True \
        --notebook-dir="$NOTEBOOKS_DIR"
else
    echo "Launching HappyTorch Web UI on ${HOST:-0.0.0.0}:${PORT:-8000}..."
    exec python start_web.py
fi
