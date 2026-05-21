import pytest
import sparse_kernel_extension
import torch

from unum_ops.sparse_kernel_extension import (
    get_block_table_ref_torch,
    get_block_table_ref_triton,
)

torch.manual_seed(42)
# decode
kHeadGroup = 2
kSparseTopK = 96
kSparseBlockSize = 64
# token_num = 1
batch_size = 4
seqlen_q_max = 8192
topk_max_val = seqlen_q_max // kSparseBlockSize


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


def gen_topk(batch_size, topk_max_val, device):
    rand_scores = torch.rand((kHeadGroup, batch_size, topk_max_val))
    indices = rand_scores.topk(kSparseTopK, dim=-1, largest=False).indices
    topk_idx = indices.sort(dim=-1).values.to(dtype=torch.int32).to(device=device)
    return topk_idx


@pytest.fixture(scope="module")
def test_data(device):
    topk_idx = gen_topk(batch_size, topk_max_val, device)
    block_table = torch.tensor(
        [_ for _ in range(1, seqlen_q_max * batch_size + 1)],
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, seqlen_q_max)
    token_to_bs = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    seqlen_q = torch.full((batch_size,), seqlen_q_max, dtype=torch.int32, device=device)

    return {
        "topk_idx": topk_idx,
        "block_table": block_table,
        "token_to_bs": token_to_bs,
        "seqlen_q": seqlen_q,
    }


@pytest.fixture(params=[1, 2, 4, 8])
def varied_batch_size(request):
    return request.param


@pytest.fixture(params=[256, 512, 1024, 2048, 4096, 8192])
def varied_seqlen(request):
    return request.param


@pytest.fixture(params=[32, 64, 96, 128])
def varied_topk(request):
    return request.param


def generate_diverse_test_data(
    batch_size_val, seqlen_val, topk_val, device, seed_offset=0
):
    topk_max_val = seqlen_val // kSparseBlockSize
    actual_topk = min(topk_val, topk_max_val)

    local_seed = (42 + seed_offset + batch_size_val + seqlen_val + topk_val) % 1000000
    gen = torch.Generator(device="cpu")
    gen.manual_seed(local_seed)

    rand_scores = torch.rand((kHeadGroup, batch_size_val, topk_max_val), generator=gen)
    indices = rand_scores.topk(actual_topk, dim=-1, largest=False).indices
    topk_idx_sorted = indices.sort(dim=-1).values

    topk_idx = torch.zeros(
        (kHeadGroup, batch_size_val, kSparseTopK), dtype=torch.int32, device=device
    )
    topk_idx[:, :, :actual_topk] = topk_idx_sorted.to(device=device)

    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(local_seed + 10000)
    block_table = torch.randint(
        1, 10000, (batch_size_val, seqlen_val), dtype=torch.int32, generator=gen2
    ).to(device=device)

    token_to_bs = torch.arange(0, batch_size_val, dtype=torch.int32, device=device)

    max_blocks = seqlen_val // kSparseBlockSize
    gen3 = torch.Generator(device="cpu")
    gen3.manual_seed(local_seed + 20000)
    seqlen_q_blocks = torch.randint(
        max_blocks // 4,
        max_blocks + 1,
        (batch_size_val,),
        dtype=torch.int32,
        generator=gen3,
    )
    seqlen_q = seqlen_q_blocks * kSparseBlockSize

    return {
        "topk_idx": topk_idx,
        "block_table": block_table,
        "token_to_bs": token_to_bs,
        "seqlen_q": seqlen_q.to(device=device),
        "batch_size": batch_size_val,
        "seqlen_val": seqlen_val,
        "topk_val": kSparseTopK,
    }


@pytest.fixture(
    scope="module",
    params=[
        {"batch": 1, "seqlen": 256, "topk": 4, "seed": 0},
        {"batch": 2, "seqlen": 512, "topk": 8, "seed": 1},
        {"batch": 4, "seqlen": 1024, "topk": 16, "seed": 2},
        {"batch": 8, "seqlen": 2048, "topk": 32, "seed": 3},
        {"batch": 4, "seqlen": 4096, "topk": 64, "seed": 4},
        {"batch": 2, "seqlen": 8192, "topk": 96, "seed": 5},
    ],
)
def diverse_test_data(request, device):
    params = request.param
    return generate_diverse_test_data(
        params["batch"], params["seqlen"], params["topk"], device, params["seed"]
    )


def test_get_block_table_v1(test_data):
    out_block_table = sparse_kernel_extension.get_block_table_v1(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    assert out_block_table.shape is not None
    assert out_block_table.device.type == "cuda"


def test_get_block_table_v2(test_data):
    out_block_table = sparse_kernel_extension.get_block_table_v2(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    assert out_block_table.shape is not None
    assert out_block_table.device.type == "cuda"


def test_get_block_table_v3(test_data):
    out_block_table = sparse_kernel_extension.get_block_table_v3(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    assert out_block_table.shape is not None
    assert out_block_table.device.type == "cuda"


def test_v2_matches_v1(test_data):
    out_v1 = sparse_kernel_extension.get_block_table_v1(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    out_v2 = sparse_kernel_extension.get_block_table_v2(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    assert torch.allclose(out_v1, out_v2), "v2 output differs from v1"


def test_v3_matches_v1(test_data):
    out_v1 = sparse_kernel_extension.get_block_table_v1(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    out_v3 = sparse_kernel_extension.get_block_table_v3(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    assert torch.allclose(out_v1, out_v3), "v3 output differs from v1"


def test_get_table_torch(test_data):
    out_v1 = sparse_kernel_extension.get_block_table_v1(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    out_torch = get_block_table_ref_torch(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
    )
    assert torch.allclose(out_v1, out_torch), "torch output differs from v1"


def test_get_table_triton(diverse_test_data):
    test_data = diverse_test_data
    # 打印shape或len
    print("\n")
    print(f'{test_data["topk_idx"].shape=}')
    print(f'{test_data["block_table"].shape=}')
    print(f'{test_data["token_to_bs"].shape=}')
    print(f'{test_data["seqlen_q"].shape=}')
    print(f'{test_data["topk_val"]=}')

    out_v1 = sparse_kernel_extension.get_block_table_v1(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        test_data["topk_val"],
    )
    out_triton = get_block_table_ref_triton(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        test_data["topk_val"],
    )
    assert torch.allclose(out_v1, out_triton), "triton output differs from v1"


def benchmark_kernel(name, func, *args, N=100):
    import time

    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        func(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000


def test_bench_get_table(test_data):
    topk_idx = test_data["topk_idx"]
    block_table = test_data["block_table"]
    token_to_bs = test_data["token_to_bs"]
    seqlen_q = test_data["seqlen_q"]

    kernels = [
        (
            "Torch",
            lambda: get_block_table_ref_torch(
                topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q
            ),
        ),
        (
            "Triton",
            lambda: get_block_table_ref_triton(
                topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q
            ),
        ),
        (
            "CUDA v1",
            lambda: sparse_kernel_extension.get_block_table_v1(
                topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q, kSparseTopK
            ),
        ),
        (
            "CUDA v2",
            lambda: sparse_kernel_extension.get_block_table_v2(
                topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q, kSparseTopK
            ),
        ),
        (
            "CUDA v3",
            lambda: sparse_kernel_extension.get_block_table_v3(
                topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q, kSparseTopK
            ),
        ),
    ]

    for _ in range(10):
        for _, func in kernels:
            func()

    times = {name: benchmark_kernel(name, func) for name, func in kernels}

    import pandas as pd

    pd.set_option("display.max_colwidth", 20)
    pd.set_option("display.width", 100)

    data = []
    for name in ["Torch", "Triton", "CUDA v1", "CUDA v2", "CUDA v3"]:
        t = times[name]
        vs_v1 = f"{t / times['CUDA v1']:.1f}x" if name != "CUDA v1" else "1.0x"
        vs_v3 = f"{t / times['CUDA v3']:.1f}x" if name != "CUDA v3" else "1.0x"
        data.append(
            {
                "Kernel": name,
                "Latency (ms)": f"{t:.3f}",
                "vs CUDA v1": vs_v1,
                "vs CUDA v3": vs_v3,
            }
        )

    df = pd.DataFrame(data)
    print("\n")
    print(df.to_string(index=False, col_space=12))
