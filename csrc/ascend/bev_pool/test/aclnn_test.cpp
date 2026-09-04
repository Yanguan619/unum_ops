#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>

#include "acl/acl.h"
#include "aclnn_bev_pool.h"

int main() {
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

    // Create test data: N=50, C=8, B=1, D=2, H=4, W=4
    int64_t N = 50, C = 8, B = 1, D = 2, H = 4, W = 4;
    int64_t gridTotal = B * D * H * W;
    // Generate random points and compute intervals
    std::vector<float> feats(N * C);
    std::vector<int32_t> coords(N * 4);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> cdist(0, (int)std::min(W, H)-1);
    int depthDist = D;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) feats[i * C + j] = fdist(rng);
        coords[i * 4 + 0] = cdist(rng) % W; // x
        coords[i * 4 + 1] = cdist(rng) % H; // y
        coords[i * 4 + 2] = rand() % D;      // z
        coords[i * 4 + 3] = 0;                // batch
    }

    // Compute ranks and sort (like Python wrapper)
    std::vector<int64_t> ranks(N);
    for (int i = 0; i < N; i++) {
        ranks[i] = (int64_t)coords[i*4+0] * (W * D * B) +
                   (int64_t)coords[i*4+1] * (D * B) +
                   (int64_t)coords[i*4+2] * B +
                   coords[i*4+3];
    }
    // argsort
    std::vector<int> indices(N);
    for (int i = 0; i < N; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return ranks[a] < ranks[b]; });

    std::vector<float> featsSorted(N * C);
    std::vector<int32_t> coordsSorted(N * 4);
    std::vector<int64_t> ranksSorted(N);
    for (int i = 0; i < N; i++) {
        int src = indices[i];
        for (int j = 0; j < C; j++) featsSorted[i * C + j] = feats[src * C + j];
        for (int j = 0; j < 4; j++) coordsSorted[i * 4 + j] = coords[src * 4 + j];
        ranksSorted[i] = ranks[src];
    }
    // interval boundaries
    std::vector<int32_t> intervalStarts, intervalLengths;
    intervalStarts.push_back(0);
    for (int i = 1; i < N; i++) {
        if (ranksSorted[i] != ranksSorted[i - 1]) {
            intervalStarts.push_back(i);
        }
    }
    intervalStarts.push_back(N);
    int K = (int)intervalStarts.size() - 1;
    for (int i = 0; i < K; i++) {
        intervalLengths.push_back(intervalStarts[i+1] - intervalStarts[i]);
    }
    intervalStarts.resize(K);
    for (int i = 0; i < K; i++) {
        fprintf(stderr, "[dbg] interval %d: start=%d length=%d\n", i, intervalStarts[i], intervalLengths[i]);
    }

    // Allocate device memory
    int64_t inShape[2] = {N, C};
    void* featsDev = nullptr;
    aclrtMalloc(&featsDev, N * C * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(featsDev, N * C * sizeof(float), featsSorted.data(), N * C * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* featsTensor = aclCreateTensor(inShape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, inShape, 2, featsDev);

    int64_t coordShape[2] = {N, 4};
    void* coordDev = nullptr;
    aclrtMalloc(&coordDev, N * 4 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(coordDev, N * 4 * sizeof(int32_t), coordsSorted.data(), N * 4 * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* coordTensor = aclCreateTensor(coordShape, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, coordShape, 2, coordDev);

    int64_t startsShape[1] = {K};
    void* startsDev = nullptr;
    aclrtMalloc(&startsDev, K * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(startsDev, K * sizeof(int32_t), intervalStarts.data(), K * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* startsTensor = aclCreateTensor(startsShape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, startsShape, 1, startsDev);

    int64_t lengthsShape[1] = {K};
    void* lengthsDev = nullptr;
    aclrtMalloc(&lengthsDev, K * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(lengthsDev, K * sizeof(int32_t), intervalLengths.data(), K * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* lengthsTensor = aclCreateTensor(lengthsShape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, lengthsShape, 1, lengthsDev);

    // output
    size_t outBytes = gridTotal * C * sizeof(float);
    void* outDev = nullptr;
    aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    int64_t outShape[5] = {B, D, H, W, C};
    aclTensor* outTensor = aclCreateTensor(outShape, 5, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, outShape, 5, outDev);

    // launch
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = aclnnBevPoolGetWorkspaceSize(
        featsTensor, coordTensor, startsTensor, lengthsTensor,
        B, D, H, W, outTensor, &workspaceSize, &executor);
    if (st != ACL_SUCCESS) {
        fprintf(stderr, "aclnnBevPoolGetWorkspaceSize failed, st=%d\n", (int)st);
        return 1;
    }
    fprintf(stderr, "[dbg] workspace=%lu K=%d\n", (unsigned long)workspaceSize, K);
    void* ws = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&ws, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    st = aclnnBevPool(ws, workspaceSize, executor, stream);
    if (st != ACL_SUCCESS) {
        fprintf(stderr, "aclnnBevPool failed, st=%d\n", (int)st);
        return 1;
    }
    fprintf(stderr, "[dbg] launched, syncing...\n");
    aclrtSynchronizeStream(stream);
    fprintf(stderr, "[dbg] sync done\n");

    // copy back
    std::vector<float> outHost(gridTotal * C);
    aclrtMemcpy(outHost.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // reference: index_add_
    std::vector<float> refHost(gridTotal * C, 0.0f);
    for (int i = 0; i < N; i++) {
        int64_t flat = (int64_t)coordsSorted[i*4+0] + (int64_t)coordsSorted[i*4+1]*W
                       + (int64_t)coordsSorted[i*4+2]*W*H + (int64_t)coordsSorted[i*4+3]*W*H*D;
        int64_t base = flat * C;
        for (int j = 0; j < C; j++) {
            refHost[base + j] += featsSorted[i*C + j];
        }
    }
    double maxDiff = 0.0;
    for (size_t i = 0; i < refHost.size(); i++) {
        double d = fabs((double)outHost[i] - (double)refHost[i]);
        if (d > maxDiff) maxDiff = d;
    }
    fprintf(stderr, "[dbg] out[0..7]=%f,%f,%f,%f,%f,%f,%f,%f\n",
            outHost[0], outHost[1], outHost[2], outHost[3],
            outHost[4], outHost[5], outHost[6], outHost[7]);
    fprintf(stderr, "[dbg] ref[0..7]=%f,%f,%f,%f,%f,%f,%f,%f\n",
            refHost[0], refHost[1], refHost[2], refHost[3],
            refHost[4], refHost[5], refHost[6], refHost[7]);

    if (maxDiff < 1e-5) {
        printf("PASS: N=%ld K=%d workspace=%lu maxDiff=%e\n", N, K, (unsigned long)workspaceSize, maxDiff);
    } else {
        printf("FAIL: N=%ld K=%d maxDiff=%e\n", N, K, maxDiff);
        return 1;
    }

    // cleanup
    if (ws) aclrtFree(ws);
    aclDestroyTensor(featsTensor); aclDestroyTensor(coordTensor);
    aclDestroyTensor(startsTensor); aclDestroyTensor(lengthsTensor); aclDestroyTensor(outTensor);
    aclrtFree(featsDev); aclrtFree(coordDev); aclrtFree(startsDev); aclrtFree(lengthsDev); aclrtFree(outDev);
    aclrtDestroyStream(stream); aclrtDestroyContext(ctx); aclrtResetDevice(devId); aclFinalize();
    return 0;
}