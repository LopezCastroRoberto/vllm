# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA attention backend."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonPrefillMetadata
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
    from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


class _FlashInferMLASparseBackendBase(AttentionBackend):
    """Common metadata for concrete FlashInfer sparse MLA backends."""

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseMetadataBuilder"]:
        return FlashInferMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class FlashInferMLASparseTRTLLMBackend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend using the TRTLLM-gen launcher."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseTRTLLMMetadataBuilder"]:
        return FlashInferMLASparseTRTLLMMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if kv_cache_dtype == "fp8_ds_mla":
            return (
                "FLASHINFER_MLA_SPARSE SM10 does not support fp8_ds_mla kv-cache dtype"
            )

        # FlashInfer MLA sparse SM10 kernel requires qk_nope_head_dim in [128, 192].
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 1)
            if qk_nope_head_dim not in [128, 192]:
                return (
                    "FlashInfer MLA Sparse kernel requires qk_nope_head_dim "
                    f"in [128, 192], but got {qk_nope_head_dim}"
                )
            # Check for index_topk which indicates sparse model
            if not hasattr(hf_text_config, "index_topk"):
                return "FlashInfer MLA Sparse requires model with index_topk config"
        return None


class FlashInferMLASparseSM120Backend(_FlashInferMLASparseBackendBase):
    """FlashInfer sparse MLA backend for SM120."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM120"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64, 256]

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
            FlashInferMLASparseSM120Impl,
        )

        return FlashInferMLASparseSM120Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        from vllm.config import get_current_vllm_config
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            return (
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API"
            )
        if dtype != torch.bfloat16:
            return "dtype not supported"
        if kv_cache_dtype not in (
            None,
            "auto",
            "fp8",
            "fp8_e4m3",
            "fp8_ds_mla",
        ):
            return "kv_cache_dtype not supported"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            index_topk = getattr(hf_text_config, "index_topk", None)
            if index_topk is None:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires a model with "
                    "index_topk config"
                )
            if int(index_topk) != 2048:
                return (
                    "FLASHINFER_MLA_SPARSE_SM120 requires index_topk=2048; "
                    f"got {index_topk}"
                )
        return None


@dataclass
class FlashInferMLASparseMetadata(AttentionMetadata):
    """Attention metadata for FlashInfer MLA Sparse backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int

    # Query start locations
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor

    # Sequence lengths for all requests (context + query)
    seq_lens: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None

    # Sparse-specific
    block_size: int = 64
    topk_tokens: int = 2048
    cp_kv_cache_interleave_size: int = 1
    physical_topk_indices: torch.Tensor | None = None
    physical_topk_valid_counts: torch.Tensor | None = None
    physical_topk_is_valid: bool = False

    # Rank-major padded query ownership and the global request block table for
    # PCP-token-sharded DCP attention. None for capture/dummy batches.
    pcp_gathered_req_id: torch.Tensor | None = None
    pcp_global_block_table: torch.Tensor | None = None

class FlashInferMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[FlashInferMLASparseMetadata]
):
    """Builder for FlashInfer MLA Sparse attention metadata."""

    metadata_cls = FlashInferMLASparseMetadata
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.physical_topk_indices = torch.empty(
            (max_num_tokens, self.topk_tokens), dtype=torch.int32, device=device
        )
        self.physical_topk_valid_counts = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )

        num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        threshold = {8: 128, 16: 128, 32: 128, 64: 256, 128: 1024}.get(
            num_q_heads, 1024
        )
        self._init_reorder_batch_threshold(
            threshold,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.physical_topk_indices = self.physical_topk_indices
        metadata.physical_topk_valid_counts = self.physical_topk_valid_counts
        return metadata


class FlashInferMLASparseTRTLLMMetadataBuilder(FlashInferMLASparseMetadataBuilder):
    """Metadata builder for the SM100 TRT-LLM sparse MLA kernel."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def _build_req_id_per_token(
        self,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> torch.Tensor:
        return common_attn_metadata.token_to_req_indices(self.req_id_per_token_buffer)


# Global workspace buffer (lazily initialized)
_fi_sparse_workspace: torch.Tensor | None = None


def _get_workspace_buffer(device: torch.device) -> torch.Tensor:
    global _fi_sparse_workspace
    if _fi_sparse_workspace is None:
        # FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace;
        # the trtllm-gen path views it as uint8, so int8 is safe for all backends.
        _fi_sparse_workspace = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.int8,
            device=device,
        )
    return _fi_sparse_workspace


class FlashInferMLASparseImpl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """FlashInfer MLA Sparse implementation.

    Uses the TRT-LLM MLA kernel with sparse_mla_top_k parameter for
    sparse attention computation.
    """

    can_return_lse_for_decode: bool = True
    lse_base_on_e: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLASparseImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLASparseImpl"
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )

        self._workspace_buffer: torch.Tensor | None = None
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None

        # fp8 query quantization is required when using fp8 kv_cache,
        # as the TRTLLM-GEN sparse MLA kernel requires matching dtypes
        # for query and kv_cache (mixed bf16+fp8 is not supported).
        self.supports_quant_query_input = True

    def _run_sparse_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices_physical: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if self._workspace_buffer is None:
                self._workspace_buffer = _get_workspace_buffer(q.device)

            if self.bmm1_scale is None:
                self.bmm1_scale = self.scale
                if is_quantized_kv_cache(self.kv_cache_dtype):
                    self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
            if self.bmm2_scale is None:
                self.bmm2_scale = 1.0
                if is_quantized_kv_cache(self.kv_cache_dtype):
                    self.bmm2_scale *= layer._k_scale_float

            from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

            # Single-token sparse decode. trtllm-gen requires the q_len_per_request
            # dim, but the sparse attention mask is fully per-token (each query token
            # carries its own top-k index row), so unsqueeze is sufficient and
            # correct. The MTP/multi-token q_len grouping is a perf-only layout and is
            # deferred until MTP is validated end-to-end for this backend.
            query = q.unsqueeze(1)
            block_tables = topk_indices_physical.unsqueeze(1)
            seq_lens_arg = seq_lens

            kernel_out = trtllm_batch_decode_with_kv_cache_mla(
                query=query,
                kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
                workspace_buffer=self._workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                block_tables=block_tables,
                seq_lens=seq_lens_arg,
                max_seq_len=attn_metadata.topk_tokens,
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                sparse_mla_top_k=attn_metadata.topk_tokens,
                return_lse=self.need_to_return_lse_for_decode,
            )
            if self.need_to_return_lse_for_decode:
                assert isinstance(kernel_out, tuple)
                o, lse = kernel_out
            else:
                assert isinstance(kernel_out, torch.Tensor)
                o = kernel_out
                lse = None

            out = o.view(-1, o.shape[-2], o.shape[-1])
            if lse is not None:
                lse = self._normalize_lse(lse, out.shape[0], out.shape[1])
                empty_rows = (topk_indices_physical == -1).all(dim=-1)
                out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
                lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
            return out, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if self.dcp_world_size > 1:
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            assert attn_metadata.physical_topk_indices is not None
            assert attn_metadata.physical_topk_valid_counts is not None
            physical_indices = attn_metadata.physical_topk_indices[:num_actual_toks]
            valid_counts = attn_metadata.physical_topk_valid_counts[:num_actual_toks]
            wrote_fresh_topk = getattr(
                layer, "indexer", None
            ) is not None and not getattr(layer, "skip_topk", False)
            if wrote_fresh_topk or not attn_metadata.physical_topk_is_valid:
                topk_indices_physical, seq_lens = (
                    triton_convert_req_index_to_global_index(
                        attn_metadata.req_id_per_token[:num_actual_toks],
                        attn_metadata.block_table,
                        topk_indices,
                        BLOCK_SIZE=attn_metadata.block_size,
                        NUM_TOPK_TOKENS=topk_indices.shape[1],
                        return_valid_counts=True,
                        output=physical_indices,
                        valid_counts_out=valid_counts,
                    )
                )
                attn_metadata.physical_topk_is_valid = True
            else:
                topk_indices_physical = physical_indices
                seq_lens = valid_counts

        return self._run_sparse_mqa_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices_physical,
            seq_lens,
            attn_metadata,
            layer,
        )

    TOKEN_SHARDED_ROWS_PER_RANK = 256

    def forward_mqa_token_sharded(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
        cp_group: GroupCoordinator,
        num_padded_tokens: int,
        w_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Attend gathered PCP query rows to each local DCP KV shard.

        Every rank gathers the same uniform, rank-major query and Top-K rows,
        computes partial sparse attention against its local KV shard, applies
        the linear V-up projection, LSE-corrects the partial output, and
        reduce-scatters token rows back to their PCP owners.
        """
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        num_actual = attn_metadata.num_actual_tokens
        assert q.shape[0] >= num_padded_tokens >= num_actual
        assert self.topk_indices_buffer is not None

        gathered_req_id = attn_metadata.pcp_gathered_req_id
        global_block_table = attn_metadata.pcp_global_block_table
        if gathered_req_id is None or global_block_table is None:
            # Capture/profiling batches have no runtime global PCP maps. Execute
            # a local shape-equivalent kernel; the eager segment is re-run with
            # real metadata at replay.
            if num_actual == 0:
                return q.new_zeros(
                    (0, q.shape[1], w_uv.shape[-1]), dtype=torch.bfloat16
                )
            out, _ = self.forward_mqa(
                q[:num_actual], kv_c_and_k_pe_cache, attn_metadata, layer
            )
            return torch.bmm(out.transpose(0, 1), w_uv).transpose(0, 1).contiguous()

        world_size = cp_group.world_size
        num_heads = q.shape[1]
        topk_local = self.topk_indices_buffer[:num_padded_tokens]
        if num_actual < num_padded_tokens:
            topk_local[num_actual:].fill_(-1)
        gathered_req_id = gathered_req_id.view(world_size, num_padded_tokens)
        out_chunks: list[torch.Tensor] = []

        for start in range(0, num_padded_tokens, self.TOKEN_SHARDED_ROWS_PER_RANK):
            stop = min(start + self.TOKEN_SHARDED_ROWS_PER_RANK, num_padded_tokens)
            rows = stop - start
            q_bytes = q[start:stop].reshape(rows, -1).view(torch.uint8)
            topk_bytes = topk_local[start:stop].contiguous().view(torch.uint8)
            packed = torch.cat((q_bytes, topk_bytes), dim=1)
            packed_all = cp_group.all_gather(packed, dim=0)
            num_all = rows * world_size
            q_all = (
                packed_all[:, : q_bytes.shape[1]]
                .contiguous()
                .view(q.dtype)
                .view(num_all, num_heads, q.shape[-1])
            )
            topk_all = (
                packed_all[:, q_bytes.shape[1] :]
                .contiguous()
                .view(torch.int32)
                .view(num_all, topk_local.shape[1])
            )
            req_all = gathered_req_id[:, start:stop].reshape(-1).contiguous()
            physical, valid_counts = triton_filter_and_convert_dcp_index(
                req_all,
                global_block_table,
                topk_all,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(
                    attn_metadata.cp_kv_cache_interleave_size
                ),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_all.shape[1],
                return_valid_counts=True,
                compact_valid_to_front=True,
            )
            partial, lse = self._run_sparse_mqa_kernel(
                q_all,
                kv_c_and_k_pe_cache,
                physical,
                valid_counts,
                attn_metadata,
                layer,
            )
            assert lse is not None
            partial = (
                torch.bmm(partial.transpose(0, 1), w_uv)
                .transpose(0, 1)
                .contiguous()
            )
            lses = cp_group.all_gather(lse.contiguous(), dim=0).view(
                world_size, num_all, num_heads
            )
            partial, _ = correct_attn_out(
                partial,
                lses,
                cp_group.rank_in_group,
                CPTritonContext(),
                is_lse_base_on_e=self.lse_base_on_e,
            )
            out_chunks.append(cp_group.reduce_scatter(partial, dim=0))

        local_out = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks)
        return local_out[:num_actual]

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        # FlashInfer returns the decode LSE either as 2D (num_tokens, num_heads)
        # or 3D ((num_tokens, num_heads, 1) / (num_tokens, 1, num_heads)).
        # Collapse all of these to the (num_tokens, num_heads) the shared DCP
        # reducer expects.
        if lse.dim() == 3:
            if lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            elif lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.shape[0] * lse.shape[1] == num_tokens:
                lse = lse.reshape(num_tokens, lse.shape[-1])
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected FlashInfer sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse
