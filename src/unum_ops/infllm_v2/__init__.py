try:
    from .infllmv2_attn_stage1_triton import infllmv2_attn_stage1_triton
    from .infllmv2_attn_stage1_triton_v2 import infllmv2_attn_stage1_triton_v2
    from .infllmv2_attn_stage1_triton_v2 import (
        infllmv2_attn_stage1_triton_v2 as infllmv2_attn_stage1,
    )
    from .max_pooling_1d_varlen import max_pooling_1d_varlen_ref_triton
    from .max_pooling_1d_varlen import (
        max_pooling_1d_varlen_ref_triton as max_pooling_1d_varlen,
    )
    _TRITON_AVAILABLE = True
except ImportError:
    infllmv2_attn_stage1_triton = None
    infllmv2_attn_stage1_triton_v2 = None
    infllmv2_attn_stage1 = None
    max_pooling_1d_varlen_ref_triton = None
    max_pooling_1d_varlen = None
    _TRITON_AVAILABLE = False

__all__ = [
    "infllmv2_attn_stage1",
    "infllmv2_attn_stage1_triton",
    "infllmv2_attn_stage1_triton_v2",
    "max_pooling_1d_varlen",
    "max_pooling_1d_varlen_ref_triton",
]
