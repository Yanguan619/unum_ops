#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TIER="${1:-fast}"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "check.sh: missing dependency '$1' (tier: $TIER)" >&2
        exit 127
    }
}

run_fast() {
    need ruff
    need pytest
    ruff check src test benchmark scripts
    ruff format --check src test benchmark scripts
    pytest test/test_registry.py -x --tb=short
}

run_full() {
    run_fast
    pytest test -x --tb=short
}

run_bench() {
    run_full
    pytest benchmark -x --tb=short -sv
    if [ -f scripts/perf_gate.py ]; then
        python scripts/perf_gate.py
    fi
}

case "$TIER" in
    fast) run_fast ;;
    full) run_full ;;
    bench) run_bench ;;
    all) run_bench ;;
    *)
        echo "usage: scripts/check.sh [fast|full|bench|all]  (default: fast)" >&2
        echo "  fast  — ruff lint/format + registry meta-test (seconds)" >&2
        echo "  full  — fast + full pytest suite" >&2
        echo "  bench — full + benchmark suite + perf gate" >&2
        exit 64
        ;;
esac

echo "check.sh [$TIER] passed"
