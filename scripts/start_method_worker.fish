#!/usr/bin/env fish
set -q LITE_ROOT; or set -gx LITE_ROOT (path resolve (path dirname (status filename))/..)
python3 "$LITE_ROOT/scripts/start_humaneval_worker.py" --lite-root "$LITE_ROOT"
