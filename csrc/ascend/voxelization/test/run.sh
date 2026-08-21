#!/bin/bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CANN=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}
CUSTOMIZE=${ASCEND_OPP_PATH:-$CANN/opp}/vendors/customize

mkdir -p "$HERE/build"
g++ -std=c++17 -O2 \
  -I"$CANN/atc/include" \
  -I"$CANN/include" \
  -I"$CUSTOMIZE/op_api/include" \
  -I"$CANN/atc/include/aclnn" \
  "$HERE/aclnn_test.cpp" \
  -L"$CUSTOMIZE/op_api/lib" -lcust_opapi \
  -L"$CANN/lib64" -lascendcl -lacl_op_compiler -lnnopbase \
  -Wl,-rpath,"$CUSTOMIZE/op_api/lib" -Wl,-rpath,"$CANN/lib64" \
  -o "$HERE/build/aclnn_test"

export LD_LIBRARY_PATH="$CUSTOMIZE/op_api/lib:$CANN/lib64:${LD_LIBRARY_PATH}"
"$HERE/build/aclnn_test" "$HERE/data"
echo "--- verify ---"
python3 "$HERE/verify.py"
