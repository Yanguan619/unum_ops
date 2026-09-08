#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
TARGET_OPP="${ASCEND_HOME}/opp/vendors/spconv"

echo "=== Clearing caches ==="
rm -rf "$SCRIPT_DIR/kernel_meta" 2>/dev/null || true
rm -rf /data/kernel_meta 2>/dev/null || true
rm -rf "$SCRIPT_DIR/build_out"
rm -rf "$SCRIPT_DIR/op_extension/build"

echo "=== Building OPP package + op_extension ==="
cd "$SCRIPT_DIR"
bash "$SCRIPT_DIR/build.sh"

PKG_DIR=$(find "$SCRIPT_DIR/build_out/_CPack_Packages" -type d -name "spconv" -path "*/vendors/spconv" 2>/dev/null | head -1)
if [ -z "$PKG_DIR" ]; then
    echo "WARNING: OPP package directory not found, skipping install"
else
    echo "=== Installing OPP into $TARGET_OPP ==="
    mkdir -p "$TARGET_OPP"
    cp -a "$PKG_DIR/." "$TARGET_OPP/"
    echo "=== OPP installed ==="

    CONFIG_PATH="${ASCEND_HOME}/opp/vendors/config.ini"
    if [ -f "$CONFIG_PATH" ] && ! grep -q "spconv" "$CONFIG_PATH"; then
        sed -i "s/^load_priority=/load_priority=spconv,/" "$CONFIG_PATH"
        echo "=== Added spconv to config.ini ==="
    fi
fi

EXT_SO="$SCRIPT_DIR/op_extension/build/libspconv_gemm_ops.so"
if [ -f "$EXT_SO" ]; then
    echo "op_extension: $EXT_SO"
    ls -la "$EXT_SO"
fi