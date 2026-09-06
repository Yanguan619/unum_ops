#!/bin/bash
# Rebuild the BevPool OPP package + op_extension .so, then install OPP into
# the CANN vendors directory so the runtime picks up the new kernel.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
TARGET_OPP="${ASCEND_HOME}/opp/vendors/customize"

echo "=== Clearing TBE compilation caches ==="
rm -rf "$SCRIPT_DIR/kernel_meta" 2>/dev/null || true
rm -rf /data/kernel_meta 2>/dev/null || true
rm -rf "$SCRIPT_DIR/build_out"
rm -rf "$SCRIPT_DIR/op_extension/build"

echo "=== Building OPP package + op_extension ==="
cd "$SCRIPT_DIR"
bash "$SCRIPT_DIR/build.sh"

# Find the generated package dynamically (platform/arch agnostic)
PKG_DIR=$(find "$SCRIPT_DIR/build_out/_CPack_Packages" -type d -name "customize" -path "*/vendors/customize" 2>/dev/null | head -1)
if [ -z "$PKG_DIR" ]; then
    echo "WARNING: OPP package directory not found, skipping install"
else
    echo "=== Installing OPP into $TARGET_OPP ==="
    mkdir -p "$TARGET_OPP"
    cp -a "$PKG_DIR/." "$TARGET_OPP/"
    echo "=== OPP installed ==="
fi

EXT_SO="$SCRIPT_DIR/op_extension/build/libbev_pool_ops.so"
echo "op_extension: $EXT_SO"
ls -la "$EXT_SO"
