#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")

python "$SCRIPT_DIR/main.py" "$@"