#!/usr/bin/env bash
# One command from a fresh copy.  Everything is local and offline.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 main.py "$@"
