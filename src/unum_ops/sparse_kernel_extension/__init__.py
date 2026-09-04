from .get_table_torch import get_block_table_ref_torch

try:
    from .get_table_triton import (
        get_block_table_ref_triton,
        get_block_table_ref_triton_v2,
        get_block_table_ref_triton_v3,
    )
    from .get_table_triton import (
        get_block_table_ref_triton as get_block_table_v2,
    )
    from .get_table_triton import (
        get_block_table_ref_triton as get_block_table_v3,
    )
    _TRITON_AVAILABLE = True
except ImportError:
    get_block_table_ref_triton = None
    get_block_table_ref_triton_v2 = None
    get_block_table_ref_triton_v3 = None
    get_block_table_v2 = None
    get_block_table_v3 = None
    _TRITON_AVAILABLE = False

__all__ = [
    "get_block_table_ref_torch",
    "get_block_table_ref_triton",
    "get_block_table_ref_triton_v2",
    "get_block_table_ref_triton_v3",
    "get_block_table_v2",
    "get_block_table_v3",
]