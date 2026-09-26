from __future__ import annotations
import argparse
import yaml
from dataclasses import dataclass
from typing import Any, Dict

from main.model.configuration_vismem import VisMemConfig, QueryBuilderConfig, LoRAConfig

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_vismem_config(cfg_dict: Dict[str, Any]) -> VisMemConfig:
    v = cfg_dict.get("vismem", cfg_dict.get("main", {}))
    qb = v.get("query_builder", {})
    lora = v.get("lora", {})
    cfg = VisMemConfig(
        short_invoke_token=v.get("short_invoke_token","<ms_I>"),
        short_end_token=v.get("short_end_token","<ms_E>"),
        long_invoke_token=v.get("long_invoke_token","<ml_I>"),
        long_end_token=v.get("long_end_token","<ml_E>"),
        query_len=int(v.get("query_len",8)),
        short_mem_len=int(v.get("short_mem_len",8)),
        long_mem_len=int(v.get("long_mem_len",16)),
        former_backend=str(v.get("former_backend","lora_llm")),
        max_prompt_hidden=int(v.get("max_prompt_hidden",1024)),
        visual_feature_source=str(v.get("visual_feature_source", "projected")),
        lvm_num_slots=int(v.get("lvm_num_slots",256)),
        lvm_segment_len=int(v.get("lvm_segment_len",32)),
        lvm_tau_r=float(v.get("lvm_tau_r",0.75)),
        lvm_tau_s=float(v.get("lvm_tau_s",0.35)),
        lvm_gate_hidden=int(v.get("lvm_gate_hidden",256)),
        lvm_normalize_slot=bool(v.get("lvm_normalize_slot", True)),
        lvm_eviction_policy=str(v.get("lvm_eviction_policy", "fifo")),
        lvm_readout_policy=str(v.get("lvm_readout_policy", "recent")),
        lvm_delta_t_scale=float(v.get("lvm_delta_t_scale", 32.0)),
        lvm_enable_graph_merge=bool(v.get("lvm_enable_graph_merge", True)),
        lvm_merge_similarity_weight=float(v.get("lvm_merge_similarity_weight", 1.0)),
        lvm_merge_surprise_weight=float(v.get("lvm_merge_surprise_weight", 0.5)),
        lvm_merge_access_weight=float(v.get("lvm_merge_access_weight", 0.25)),
        lvm_merge_recency_weight=float(v.get("lvm_merge_recency_weight", 0.25)),
        lvm_segment_encoder_layers=int(v.get("lvm_segment_encoder_layers", 1)),
        lvm_segment_encoder_heads=int(v.get("lvm_segment_encoder_heads", 4)),
        lvm_segment_time_pos=bool(v.get("lvm_segment_time_pos", True)),
        lvm_use_saw=bool(v.get("lvm_use_saw", False)),
        lvm_saw_lambda_attn=float(v.get("lvm_saw_lambda_attn", 0.5)),
        lvm_saw_ema_decay=float(v.get("lvm_saw_ema_decay", 0.9)),
        lvm_saw_l_min=int(v.get("lvm_saw_l_min", 8)),
        lvm_saw_l_max=int(v.get("lvm_saw_l_max", 64)),
        lvm_saw_energy_budget=float(v.get("lvm_saw_energy_budget", 8.0)),
        lvm_saw_recent_window=int(v.get("lvm_saw_recent_window", 64)),
        lvm_saw_quantile=float(v.get("lvm_saw_quantile", 0.9)),
        lvm_saw_attn_groups=int(v.get("lvm_saw_attn_groups", 32)),
        lvm_saw_use_attn_proxy=bool(v.get("lvm_saw_use_attn_proxy", True)),
        method_enable_dmg_gar=bool(v.get("method_enable_dmg_gar", True)),
        method_graph_budget=int(v.get("method_graph_budget", 64)),
        method_seed_topk=int(v.get("method_seed_topk", 16)),
        method_num_hops=int(v.get("method_num_hops", 2)),
        method_evidence_tokens=int(v.get("method_evidence_tokens", 8)),
        method_eta_surprise_boost=float(v.get("method_eta_surprise_boost", 0.2)),
        method_span_penalty_weight=float(v.get("method_span_penalty_weight", 0.05)),
        method_merge_penalty_weight=float(v.get("method_merge_penalty_weight", 0.05)),
        method_route_edge_weight=float(v.get("method_route_edge_weight", 0.1)),
        method_route_temperature=float(v.get("method_route_temperature", 1.0)),
        method_route_threshold=float(v.get("method_route_threshold", 0.0)),
        method_sim_topk=int(v.get("method_sim_topk", 4)),
        method_temporal_weight_c=float(v.get("method_temporal_weight_c", 1.0)),
        method_gar_layers=int(v.get("method_gar_layers", 2)),
        method_spar_subgraph_weight=float(v.get("method_spar_subgraph_weight", 1.0)),
        method_spar_redundancy_weight=float(v.get("method_spar_redundancy_weight", 1.0)),
        method_redundancy_margin=float(v.get("method_redundancy_margin", 0.0)),
        query_builder=QueryBuilderConfig(
            num_layers=int(qb.get("num_layers",2)),
            num_heads=int(qb.get("num_heads",8)),
            dropout=float(qb.get("dropout",0.0)),
            ff_mult=int(qb.get("ff_mult",4)),
        ),
        lora=LoRAConfig(
            r=int(lora.get("r",16)),
            alpha=int(lora.get("alpha",32)),
            dropout=float(lora.get("dropout",0.05)),
            target_modules = list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            short_target_modules = lora.get("short_target_modules", None),
            long_target_modules = lora.get("long_target_modules", None),
        ),
    )
    return cfg
