#!/bin/bash
# Rebuild the BevPool OPP package + op_extension .so, then install OPP into
# the CANN vendors directory so the runtime picks up the new kernel.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_OPP=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize

echo "=== Clearing TBE compilation caches ==="
rm -rf /data/kernel_meta /workspace/unum_ops/kernel_meta
rm -rf "$SCRIPT_DIR/build_out"

echo "=== Building OPP package + op_extension ==="
bash "$SCRIPT_DIR/build.sh"

PKG_DIR="$SCRIPT_DIR/build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize"
if [ -d "$PKG_DIR" ]; then
    echo "=== Installing OPP into $TARGET_OPP ==="
    # Clean previous customize deployment of bev_pool ops (keep op_api header dir intact by overwrite)
    cp -a "$PKG_DIR/." "$TARGET_OPP/"
    echo "=== OPP installed ==="
fi

EXT_SO="$SCRIPT_DIR/op_extension/build/libbev_pool_ops.so"
echo "op_extension: $EXT_SO"
ls -la "$EXT_SO"
