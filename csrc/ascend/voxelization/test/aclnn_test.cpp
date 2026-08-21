#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "acl/acl.h"
#include "aclnn_voxelization.h"

constexpr int64_t MAX_VOXELS = 40000;
constexpr int64_t MAX_POINTS_PER_VOXEL = 32;
constexpr int64_t MAX_NUM_POINTS = 3000000;

static bool ReadFile(const std::string& path, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    data.resize(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return true;
}

static void WriteFile(const std::string& path, const void* p, size_t nbytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), nbytes);
}

int main(int argc, char** argv) {
    const char* dataDir = (argc > 1) ? argv[1] : "test/data";
    int64_t maxVoxels = (argc > 2) ? atoll(argv[2]) : MAX_VOXELS;

    std::string inDir = std::string(dataDir) + "/input";
    std::string outDir = std::string(dataDir) + "/output";

    std::vector<float> points;
    if (!ReadFile(inDir + "/points.bin", points)) return 1;
    int64_t N = points.size() / 4;

    float voxelSize[3] = {0.16f, 0.16f, 4.0f};
    float pcr[6] = {0.0f, -39.68f, -3.0f, 69.12f, 39.68f, 1.0f};

    // acl init
    if (aclInit(nullptr) != ACL_SUCCESS) {
        fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    int32_t devId = 0;
    aclrtContext ctx = nullptr;
    aclrtStream stream = nullptr;
    if (aclrtSetDevice(devId) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetDevice failed\n");
        return 1;
    }
    aclrtCreateContext(&ctx, devId);
    aclrtCreateStream(&stream);

    // input tensor (N,4)
    int64_t inShape[2] = {N, 4};
    int64_t inStrides[2] = {4, 1};
    void* inDev = nullptr;
    aclrtMalloc(&inDev, N * 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inDev, N * 4 * sizeof(float), points.data(), N * 4 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* pointsTensor = aclCreateTensor(
        inShape, 2, ACL_FLOAT, inStrides, ACL_FORMAT_ND, ACL_FORMAT_ND, inShape, 2, inDev);

    // output tensors (full capacity buffers)
    size_t voxBytes = maxVoxels * MAX_POINTS_PER_VOXEL * 4 * sizeof(float);
    size_t coordBytes = maxVoxels * 3 * sizeof(int32_t);
    size_t nptsBytes = maxVoxels * sizeof(int32_t);
    size_t nvoxBytes = 64;  // 增大读取范围以检测偏移

    void* voxDev = nullptr;
    void* coordDev = nullptr;
    void* nptsDev = nullptr;
    void* nvoxDev = nullptr;
    aclrtMalloc(&voxDev, voxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&coordDev, coordBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&nptsDev, nptsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&nvoxDev, nvoxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    int64_t voxShape[3] = {maxVoxels, MAX_POINTS_PER_VOXEL, 4};
    int64_t voxStrides[3] = {128, 4, 1};
    int64_t coordShape[2] = {maxVoxels, 3};
    int64_t coordStrides[2] = {3, 1};
    int64_t nptsShape[1] = {maxVoxels};
    int64_t nptsStrides[1] = {1};
    int64_t nvoxShape[1] = {1};
    int64_t nvoxStrides[1] = {1};

    aclTensor* voxTensor = aclCreateTensor(
        voxShape, 3, ACL_FLOAT, voxStrides, ACL_FORMAT_ND, ACL_FORMAT_ND, voxShape, 3, voxDev);
    aclTensor* coordTensor = aclCreateTensor(
        coordShape, 2, ACL_INT32, coordStrides, ACL_FORMAT_ND, ACL_FORMAT_ND, coordShape, 2, coordDev);
    aclTensor* nptsTensor = aclCreateTensor(
        nptsShape, 1, ACL_INT32, nptsStrides, ACL_FORMAT_ND, ACL_FORMAT_ND, nptsShape, 1, nptsDev);
    aclTensor* nvoxTensor = aclCreateTensor(
        nvoxShape, 1, ACL_INT32, nvoxStrides, ACL_FORMAT_ND, ACL_FORMAT_ND, nvoxShape, 1, nvoxDev);

    aclFloatArray* voxSizeArr = aclCreateFloatArray(voxelSize, 3);
    aclFloatArray* pcrArr = aclCreateFloatArray(pcr, 6);

    // launch
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = aclnnVoxelizationGetWorkspaceSize(
        pointsTensor, voxSizeArr, pcrArr, MAX_POINTS_PER_VOXEL, maxVoxels,
        voxTensor, coordTensor, nptsTensor, nvoxTensor, &workspaceSize, &executor);
    if (st != ACL_SUCCESS) {
        fprintf(stderr, "aclnnVoxelizationGetWorkspaceSize failed, st=%d\n", (int)st);
        return 1;
    }
    fprintf(stderr, "[dbg] workspace=%lu\n", (unsigned long)workspaceSize);
    void* ws = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    st = aclnnVoxelization(ws, workspaceSize, executor, stream);
    if (st != ACL_SUCCESS) {
        fprintf(stderr, "aclnnVoxelization failed, st=%d\n", (int)st);
        return 1;
    }
    fprintf(stderr, "[dbg] aclnnVoxelization launched, syncing...\n");
    aclrtSynchronizeStream(stream);
    fprintf(stderr, "[dbg] sync done\n");

    // LAUNCHES=N: 额外 launch 次数（warmup=0, bench=N）；默认 3 warmup + 20 bench
    const char* launchesStr = getenv("LAUNCHES");
    int extraLaunches = 20;
    if (launchesStr) {
        extraLaunches = atoi(launchesStr);
        fprintf(stderr, "[dbg] LAUNCHES=%d\n", extraLaunches);
    }
    int warm = (launchesStr ? 0 : 3);
    for (int w = 0; w < warm; w++) {
        aclnnVoxelization(ws, workspaceSize, executor, stream);
    }
    aclrtSynchronizeStream(stream);

    // benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < extraLaunches; iter++) {
        aclnnVoxelization(ws, workspaceSize, executor, stream);
    }
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avgMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / extraLaunches;
    fprintf(stderr, "[bench] %d iters avg: %.2f ms/iter\n", extraLaunches, avgMs);

    // copy back (kernel subtracts 8-byte framework header internally, so read from base)
    const int64_t OUT_HDR = 0;
    std::vector<float> voxHost(maxVoxels * MAX_POINTS_PER_VOXEL * 4);
    std::vector<int32_t> coordHost(maxVoxels * 3);
    std::vector<int32_t> nptsHost(maxVoxels);
    int32_t nvoxHost[16] = {0};
    aclrtMemcpy(voxHost.data(), voxBytes, (char*)voxDev, voxBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(coordHost.data(), coordBytes, (char*)coordDev, coordBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(nptsHost.data(), nptsBytes, (char*)nptsDev, nptsBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(nvoxHost, 64, (char*)nvoxDev, 56, ACL_MEMCPY_DEVICE_TO_HOST);

    // ---- final nvox ----
    fprintf(stderr, "[dbg] nvox[0..3]: [%d, %d, %d, %d]\n",
            nvoxHost[0], nvoxHost[1], nvoxHost[2], nvoxHost[3]);

    WriteFile(outDir + "/voxels.bin", voxHost.data(), (size_t)nvoxHost[0] * 32 * 4 * sizeof(float));
    WriteFile(outDir + "/coords.bin", coordHost.data(), (size_t)nvoxHost[0] * 3 * sizeof(int32_t));
    WriteFile(outDir + "/num_points.bin", nptsHost.data(), (size_t)nvoxHost[0] * sizeof(int32_t));
    std::ofstream f(outDir + "/num_voxels.txt");
    f << nvoxHost[0] << "\n";

    printf("N=%ld workspace=%lu kernel_M=%d voxels[0..1]=[%g,%g]\n",
           N, (unsigned long)workspaceSize, nvoxHost[0],
           voxHost[0], voxHost[1]);

    if (ws) aclrtFree(ws);
    aclDestroyTensor(pointsTensor);
    aclDestroyTensor(voxTensor);
    aclDestroyTensor(coordTensor);
    aclDestroyTensor(nptsTensor);
    aclDestroyTensor(nvoxTensor);
    aclDestroyFloatArray(voxSizeArr);
    aclDestroyFloatArray(pcrArr);
    aclrtFree(inDev);
    aclrtFree(voxDev);
    aclrtFree(coordDev);
    aclrtFree(nptsDev);
    aclrtFree(nvoxDev);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(devId);
    aclFinalize();
    return 0;
}
