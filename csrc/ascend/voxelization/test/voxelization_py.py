"""Ascend 310P Voxelization — Python 调用（ctypes 封装 aclnn API）"""
import ctypes, os, numpy as np

# ---------- 加载 ACL 运行时和自定义算子 .so ----------
ACL_HOME = "/usr/local/Ascend/cann-9.0.0/aarch64-linux"
OPAPI_LIB = "/usr/local/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib/libcust_opapi.so"

# 必须先加载 libnnopbase.so（包含 aclCreateTensor 等），再加载 libcust_opapi.so
_nnop = ctypes.CDLL(os.path.join(ACL_HOME, "lib64/libnnopbase.so"), ctypes.RTLD_GLOBAL)
_acl = ctypes.CDLL(os.path.join(ACL_HOME, "lib64/libascendcl.so"), ctypes.RTLD_GLOBAL)
_cust = ctypes.CDLL(OPAPI_LIB, ctypes.RTLD_GLOBAL)

# ACL 类型
aclTensor = ctypes.c_void_p
aclFloatArray = ctypes.c_void_p
aclOpExecutor = ctypes.c_void_p
aclrtStream = ctypes.c_void_p

# ---------- 定义函数签名 ----------
def _bind(lib, name, argtypes, restype):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn

# acl 初始化
_bind(_acl, "aclInit", [ctypes.c_void_p], ctypes.c_int32)
_bind(_acl, "aclrtSetDevice", [ctypes.c_int32], ctypes.c_int32)
_bind(_acl, "aclrtCreateContext", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32], ctypes.c_int32)
_bind(_acl, "aclrtCreateStream", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int32)
_bind(_acl, "aclrtMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int32], ctypes.c_int32)
_bind(_acl, "aclrtMemcpy", [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32], ctypes.c_int32)
_bind(_acl, "aclrtSynchronizeStream", [ctypes.c_void_p], ctypes.c_int32)
_bind(_acl, "aclrtFree", [ctypes.c_void_p], ctypes.c_int32)
_bind(_acl, "aclrtDestroyStream", [ctypes.c_void_p], ctypes.c_int32)
_bind(_acl, "aclrtDestroyContext", [ctypes.c_void_p], ctypes.c_int32)
_bind(_acl, "aclrtResetDevice", [ctypes.c_int32], ctypes.c_int32)

# acl tensor / array 函数（在 libnnopbase.so 中）
_bind(_nnop, "aclCreateTensor", [ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32,
                                 ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32,
                                 ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_void_p], ctypes.c_void_p)
_bind(_nnop, "aclDestroyTensor", [ctypes.c_void_p], ctypes.c_int32)
_bind(_nnop, "aclCreateFloatArray", [ctypes.POINTER(ctypes.c_float), ctypes.c_int64], ctypes.c_void_p)
_bind(_nnop, "aclDestroyFloatArray", [ctypes.c_void_p], ctypes.c_int32)

# 算子 API
_bind(_cust, "aclnnVoxelizationGetWorkspaceSize", [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)
], ctypes.c_int32)

_bind(_cust, "aclnnVoxelization", [
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
], ctypes.c_int32)


# ---------- 全局初始化 ----------
_ctx = ctypes.c_void_p()
_stream = ctypes.c_void_p()
_acl.aclInit(None)
_acl.aclrtSetDevice(0)
_acl.aclrtCreateContext(ctypes.byref(_ctx), 0)
_acl.aclrtCreateStream(ctypes.byref(_stream))
_initialized = True


def voxelization(points_np, voxel_size=(0.16, 0.16, 4.0), pcr=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
                 max_voxels=40000, max_points_per_voxel=32):
    """调用 Ascend 310P Voxelization 算子

    Args:
        points_np: (N, 4) float32 numpy array
        voxel_size: (vx, vy, vz)
        pcr: (x_min, y_min, z_min, x_max, y_max, z_max)
        max_voxels: 最大 voxel 数
        max_points_per_voxel: 每 voxel 最大点数

    Returns:
        voxels: (M, max_points_per_voxel, 4) float32
        coords: (M, 3) int32
        num_points: (M,) int32
        num_voxels: scalar int32
    """
    N = points_np.shape[0]
    points_np = np.ascontiguousarray(points_np, dtype=np.float32)

    # 输入 tensor
    in_shape = (ctypes.c_int64 * 2)(N, 4)
    in_strides = (ctypes.c_int64 * 2)(4, 1)
    in_dev = ctypes.c_void_p()
    _acl.aclrtMalloc(ctypes.byref(in_dev), N * 4 * 4, 1)
    _acl.aclrtMemcpy(in_dev, N * 4 * 4, points_np.ctypes.data_as(ctypes.c_void_p), N * 4 * 4, 1)
    points_tensor = _nnop.aclCreateTensor(in_shape, 2, 0, in_strides, 0, 0, in_shape, 2, in_dev)

    # 输出 tensor
    vox_bytes = max_voxels * max_points_per_voxel * 4 * 4
    coord_bytes = max_voxels * 3 * 4
    npts_bytes = max_voxels * 4
    nvox_bytes = 64

    vox_dev = ctypes.c_void_p(); coord_dev = ctypes.c_void_p()
    npts_dev = ctypes.c_void_p(); nvox_dev = ctypes.c_void_p()
    _acl.aclrtMalloc(ctypes.byref(vox_dev), vox_bytes, 1)
    _acl.aclrtMalloc(ctypes.byref(coord_dev), coord_bytes, 1)
    _acl.aclrtMalloc(ctypes.byref(npts_dev), npts_bytes, 1)
    _acl.aclrtMalloc(ctypes.byref(nvox_dev), nvox_bytes, 1)

    vox_shape = (ctypes.c_int64 * 3)(max_voxels, max_points_per_voxel, 4)
    vox_strides = (ctypes.c_int64 * 3)(max_points_per_voxel * 4, 4, 1)
    coord_shape = (ctypes.c_int64 * 2)(max_voxels, 3)
    coord_strides = (ctypes.c_int64 * 2)(3, 1)
    npts_shape = (ctypes.c_int64 * 1)(max_voxels)
    npts_strides = (ctypes.c_int64 * 1)(1)
    nvox_shape = (ctypes.c_int64 * 1)(1)
    nvox_strides = (ctypes.c_int64 * 1)(1)

    vox_tensor = _nnop.aclCreateTensor(vox_shape, 3, 0, vox_strides, 0, 0, vox_shape, 3, vox_dev)
    coord_tensor = _nnop.aclCreateTensor(coord_shape, 2, 3, coord_strides, 0, 0, coord_shape, 2, coord_dev)
    npts_tensor = _nnop.aclCreateTensor(npts_shape, 1, 3, npts_strides, 0, 0, npts_shape, 1, npts_dev)
    nvox_tensor = _nnop.aclCreateTensor(nvox_shape, 1, 3, nvox_strides, 0, 0, nvox_shape, 1, nvox_dev)

    # 参数数组
    voxel_size_arr = _nnop.aclCreateFloatArray((ctypes.c_float * 3)(*voxel_size), 3)
    pcr_arr = _nnop.aclCreateFloatArray((ctypes.c_float * 6)(*pcr), 6)

    # 获取 workspace + executor
    ws_size = ctypes.c_uint64()
    executor = ctypes.c_void_p()
    _cust.aclnnVoxelizationGetWorkspaceSize(points_tensor, voxel_size_arr, pcr_arr,
                       ctypes.c_int64(max_points_per_voxel), ctypes.c_int64(max_voxels),
                       vox_tensor, coord_tensor, npts_tensor, nvox_tensor,
                       ctypes.byref(ws_size), ctypes.byref(executor))

    ws_dev = ctypes.c_void_p()
    if ws_size.value > 0:
        _acl.aclrtMalloc(ctypes.byref(ws_dev), ws_size.value, 1)

    # 执行
    _cust.aclnnVoxelization(ws_dev, ws_size.value, executor, _stream)
    _acl.aclrtSynchronizeStream(_stream)

    # 读回结果
    nvox_host = (ctypes.c_int32 * 16)()
    _acl.aclrtMemcpy(nvox_host, 64, nvox_dev, 64, 2)
    M = nvox_host[0]

    if M > 0 and M <= max_voxels:
        vox_host = np.empty((M, max_points_per_voxel, 4), dtype=np.float32)
        coord_host = np.empty((M, 3), dtype=np.int32)
        npts_host = np.empty(M, dtype=np.int32)
        _acl.aclrtMemcpy(vox_host.ctypes.data_as(ctypes.c_void_p), M * max_points_per_voxel * 4 * 4, vox_dev, vox_bytes, 2)
        _acl.aclrtMemcpy(coord_host.ctypes.data_as(ctypes.c_void_p), M * 3 * 4, coord_dev, coord_bytes, 2)
        _acl.aclrtMemcpy(npts_host.ctypes.data_as(ctypes.c_void_p), M * 4, npts_dev, npts_bytes, 2)
    else:
        vox_host = np.empty((0, max_points_per_voxel, 4), dtype=np.float32)
        coord_host = np.empty((0, 3), dtype=np.int32)
        npts_host = np.empty(0, dtype=np.int32)

    # 释放
    _acl.aclrtFree(in_dev)
    _acl.aclrtFree(vox_dev); _acl.aclrtFree(coord_dev)
    _acl.aclrtFree(npts_dev); _acl.aclrtFree(nvox_dev)
    if ws_size.value > 0: _acl.aclrtFree(ws_dev)
    _nnop.aclDestroyTensor(points_tensor)
    _nnop.aclDestroyTensor(vox_tensor); _nnop.aclDestroyTensor(coord_tensor)
    _nnop.aclDestroyTensor(npts_tensor); _nnop.aclDestroyTensor(nvox_tensor)
    _nnop.aclDestroyFloatArray(voxel_size_arr); _nnop.aclDestroyFloatArray(pcr_arr)

    return vox_host, coord_host, npts_host, M


if __name__ == "__main__":
    # 测试
    data = np.fromfile("test/data/input/points.bin", dtype=np.float32).reshape(-1, 4)
    print(f"points: {data.shape}")
    vox, coord, npts, M = voxelization(data)
    print(f"num_voxels={M}, voxels.shape={vox.shape}")
    import sys; sys.exit(0)