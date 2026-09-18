#!/bin/bash
# Rebuild the BevPoolV3 OPP package + op_extension .so, then install OPP into
# the CANN vendors directory so the runtime picks up the new kernel.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
# 包 vendor 名为 bevpoolV3（见 custom_op.json + CMakePresets.json）。
VENDOR_NAME="bevpoolV3"
# 与 unum_ops bev_pool 错开：装到独立子目录，避免顶层级联加载冲突。
TARGET_OPP="${ASCEND_HOME}/opp/vendors/${VENDOR_NAME}"

echo "=== Clearing TBE compilation caches ==="
rm -rf "$SCRIPT_DIR/kernel_meta" 2>/dev/null || true
rm -rf /data/kernel_meta 2>/dev/null || true
rm -rf "$SCRIPT_DIR/build_out"
rm -rf "$SCRIPT_DIR/op_extension/build"

echo "=== Building OPP package + op_extension ==="
cd "$SCRIPT_DIR"
bash "$SCRIPT_DIR/build.sh"

# Find the generated package dynamically. vendor 目录命名由 package_name=${vendor_name} 决定。
PKG_DIR=$(find "$SCRIPT_DIR/build_out/_CPack_Packages" -type d -name "${VENDOR_NAME}" -path "*/vendors/${VENDOR_NAME}" 2>/dev/null | head -1)
if [ -z "$PKG_DIR" ]; then
    echo "WARNING: OPP package directory (vendors/${VENDOR_NAME}) not found, skipping install"
else
    echo "=== Installing OPP into $TARGET_OPP ==="
    mkdir -p "$TARGET_OPP"
    cp -a "$PKG_DIR/." "$TARGET_OPP/"
    echo "=== OPP installed ==="
fi

EXT_SO="$SCRIPT_DIR/op_extension/build/libbevpoolV3_ops.so"
echo "op_extension: $EXT_SO"
ls -la "$EXT_SO" 2>/dev/null || echo "(not built)"
