from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class QueryBuilderConfig:
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    ff_mult: int = 4

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj","k_proj","v_proj","o_proj"])
    short_target_modules: List[str] | None = None
    long_target_modules: List[str] | None = None

@dataclass
class VisMemConfig:
    short_invoke_token: str = "<ms_I>"
    short_end_token: str = "<ms_E>"
    long_invoke_token: str = "<ml_I>"
    long_end_token: str = "<ml_E>"

    query_len: int = 8
    short_mem_len: int = 8
    long_mem_len: int = 16

    query_builder: QueryBuilderConfig = field(default_factory=QueryBuilderConfig)

    former_backend: str = "lora_llm"   # lora_llm | tiny_transformer | lvm_sft
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    max_prompt_hidden: int = 1024   # cap to avoid huge query inputs

    # Memory is written from projected visual embeddings Phi_v(x_t) ("projected", paper default)
    # or from last-layer decoder states ("last_hidden", legacy).
    visual_feature_source: str = "projected"

    # LVM-SFT writer (used when former_backend == "lvm_sft")
    lvm_num_slots: int = 256
    lvm_segment_len: int = 32
    lvm_tau_r: float = 0.75
    lvm_tau_s: float = 0.35
    lvm_gate_hidden: int = 256
    lvm_normalize_slot: bool = True
    lvm_eviction_policy: str = "fifo"      # fifo | low_surprise | low_access
    lvm_readout_policy: str = "recent"     # recent | surprise | hybrid
    lvm_delta_t_scale: float = 32.0
    lvm_enable_graph_merge: bool = True
    lvm_merge_similarity_weight: float = 1.0
    lvm_merge_surprise_weight: float = 0.5
    lvm_merge_access_weight: float = 0.25
    lvm_merge_recency_weight: float = 0.25
    lvm_segment_encoder_layers: int = 1
    lvm_segment_encoder_heads: int = 4
    lvm_segment_time_pos: bool = True

    # SAW (Surprise-driven Adaptive Windowing)
    lvm_use_saw: bool = False
    lvm_saw_lambda_attn: float = 0.5
    lvm_saw_ema_decay: float = 0.9
    lvm_saw_l_min: int = 8
    lvm_saw_l_max: int = 64
    lvm_saw_energy_budget: float = 8.0
    lvm_saw_recent_window: int = 64
    lvm_saw_quantile: float = 0.9
    lvm_saw_attn_groups: int = 32
    lvm_saw_use_attn_proxy: bool = True

    # SelectStream: DMG + GAR
    method_enable_dmg_gar: bool = True
    method_graph_budget: int = 64
    method_seed_topk: int = 16
    method_num_hops: int = 2
    method_evidence_tokens: int = 8
    method_eta_surprise_boost: float = 0.2
    method_span_penalty_weight: float = 0.05
    method_merge_penalty_weight: float = 0.05
    method_route_edge_weight: float = 0.1
    method_route_temperature: float = 1.0
    method_route_threshold: float = 0.0
    method_sim_topk: int = 4
    method_temporal_weight_c: float = 1.0
    method_gar_layers: int = 2
    method_spar_subgraph_weight: float = 1.0
    method_spar_redundancy_weight: float = 1.0
    method_redundancy_margin: float = 0.0
