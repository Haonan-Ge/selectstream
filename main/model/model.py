from __future__ import annotations
import math
from dataclasses import asdict
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from main.model.configuration_vismem import VisMemConfig
from main.model.query_builder import QueryBuilder
from main.model.memory_former import TinyMemoryFormer
from main.model.lvm_memory import LVMMemoryFormer, LVMMemoryState
from main.model.dmg_gar import DynamicMemoryGraphReasoner
from main.model.lora_utils import is_peft_available, make_lora_adapters, set_active_adapter

class VisMemModel(nn.Module):


    def __init__(self, base_model, tokenizer, processor, config: VisMemConfig):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.cfg = config

        # Identify hidden size
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None and hasattr(base_model.config, "text_config"):
            hidden_size = getattr(base_model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from base_model.config.")

        self.hidden_size = hidden_size

        # Query builder
        qb = config.query_builder
        self.query_builder = QueryBuilder(
            hidden_size=hidden_size,
            query_len=config.query_len,
            num_layers=qb.num_layers,
            num_heads=qb.num_heads,
            dropout=qb.dropout,
            ff_mult=qb.ff_mult,
        )

        # Memory formers
        self.former_backend = config.former_backend
        self.short_former = None
        self.long_former = None
        self.lvm_short_former = None
        self.lvm_long_former = None
        self.graph_reasoner = None

        if self.former_backend == "lvm_sft":
            self.peft_model = None
            self.lvm_short_former = LVMMemoryFormer(
                hidden_size=hidden_size,
                mem_len=config.short_mem_len,
                num_slots=config.lvm_num_slots,
                segment_len=config.lvm_segment_len,
                tau_r=config.lvm_tau_r,
                tau_s=config.lvm_tau_s,
                gate_hidden=config.lvm_gate_hidden,
                normalize_slot=config.lvm_normalize_slot,
                eviction_policy=config.lvm_eviction_policy,
                readout_policy=config.lvm_readout_policy,
                delta_t_scale=config.lvm_delta_t_scale,
                enable_graph_merge=config.lvm_enable_graph_merge,
                merge_similarity_weight=config.lvm_merge_similarity_weight,
                merge_surprise_weight=config.lvm_merge_surprise_weight,
                merge_access_weight=config.lvm_merge_access_weight,
                merge_recency_weight=config.lvm_merge_recency_weight,
                segment_encoder_layers=config.lvm_segment_encoder_layers,
                segment_encoder_heads=config.lvm_segment_encoder_heads,
                segment_time_pos=config.lvm_segment_time_pos,
                use_saw=config.lvm_use_saw,
                saw_lambda_attn=config.lvm_saw_lambda_attn,
                saw_ema_decay=config.lvm_saw_ema_decay,
                saw_l_min=config.lvm_saw_l_min,
                saw_l_max=config.lvm_saw_l_max,
                saw_energy_budget=config.lvm_saw_energy_budget,
                saw_recent_window=config.lvm_saw_recent_window,
                saw_quantile=config.lvm_saw_quantile,
                saw_attn_groups=config.lvm_saw_attn_groups,
                saw_use_attn_proxy=config.lvm_saw_use_attn_proxy,
                graph_temporal_weight_c=config.method_temporal_weight_c,
                graph_sim_topk=config.method_sim_topk,
            )
            self.lvm_long_former = LVMMemoryFormer(
                hidden_size=hidden_size,
                mem_len=config.long_mem_len,
                num_slots=config.lvm_num_slots,
                segment_len=config.lvm_segment_len,
                tau_r=config.lvm_tau_r,
                tau_s=config.lvm_tau_s,
                gate_hidden=config.lvm_gate_hidden,
                normalize_slot=config.lvm_normalize_slot,
                eviction_policy=config.lvm_eviction_policy,
                readout_policy=config.lvm_readout_policy,
                delta_t_scale=config.lvm_delta_t_scale,
                enable_graph_merge=config.lvm_enable_graph_merge,
                merge_similarity_weight=config.lvm_merge_similarity_weight,
                merge_surprise_weight=config.lvm_merge_surprise_weight,
                merge_access_weight=config.lvm_merge_access_weight,
                merge_recency_weight=config.lvm_merge_recency_weight,
                segment_encoder_layers=config.lvm_segment_encoder_layers,
                segment_encoder_heads=config.lvm_segment_encoder_heads,
                segment_time_pos=config.lvm_segment_time_pos,
                use_saw=config.lvm_use_saw,
                saw_lambda_attn=config.lvm_saw_lambda_attn,
                saw_ema_decay=config.lvm_saw_ema_decay,
                saw_l_min=config.lvm_saw_l_min,
                saw_l_max=config.lvm_saw_l_max,
                saw_energy_budget=config.lvm_saw_energy_budget,
                saw_recent_window=config.lvm_saw_recent_window,
                saw_quantile=config.lvm_saw_quantile,
                saw_attn_groups=config.lvm_saw_attn_groups,
                saw_use_attn_proxy=config.lvm_saw_use_attn_proxy,
                graph_temporal_weight_c=config.method_temporal_weight_c,
                graph_sim_topk=config.method_sim_topk,
            )
            if config.method_enable_dmg_gar:
                self.graph_reasoner = DynamicMemoryGraphReasoner(
                    hidden_size=hidden_size,
                    graph_budget=config.method_graph_budget,
                    seed_topk=config.method_seed_topk,
                    num_hops=config.method_num_hops,
                    evidence_tokens=config.method_evidence_tokens,
                    eta_surprise_boost=config.method_eta_surprise_boost,
                    span_penalty_weight=config.method_span_penalty_weight,
                    merge_penalty_weight=config.method_merge_penalty_weight,
                    route_edge_weight=config.method_route_edge_weight,
                    route_temperature=config.method_route_temperature,
                    route_threshold=config.method_route_threshold,
                    sim_topk=config.method_sim_topk,
                    temporal_weight_c=config.method_temporal_weight_c,
                    gar_layers=config.method_gar_layers,
                    spar_subgraph_weight=config.method_spar_subgraph_weight,
                    spar_redundancy_weight=config.method_spar_redundancy_weight,
                    redundancy_margin=config.method_redundancy_margin,
                )
        elif self.former_backend == "tiny_transformer" or not is_peft_available():
            self.short_former = TinyMemoryFormer(hidden_size, config.short_mem_len, num_layers=2, num_heads=8)
            self.long_former  = TinyMemoryFormer(hidden_size, config.long_mem_len,  num_layers=2, num_heads=8)
            self.peft_model = None
        elif self.former_backend == "lora_llm":
            lora = config.lora
            short_targets = lora.short_target_modules or lora.target_modules
            long_targets = lora.long_target_modules or lora.target_modules
            #
            self.peft_model = make_lora_adapters(base_model, "short_former", lora.r, lora.alpha, lora.dropout, short_targets)
            from peft import LoraConfig
            self.peft_model.add_adapter(
                "long_former",
                LoraConfig(
                    r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                    bias="none", task_type="CAUSAL_LM", target_modules=long_targets
                )
            )
            self.m_init_short = nn.Parameter(torch.randn(1, config.short_mem_len, hidden_size) * 0.02)
            self.m_init_long = nn.Parameter(torch.randn(1, config.long_mem_len, hidden_size) * 0.02)
        else:
            raise ValueError(f"Unknown former_backend: {self.former_backend}")

        # Token ids
        self.short_invoke_id = tokenizer.convert_tokens_to_ids(config.short_invoke_token)
        self.short_end_id    = tokenizer.convert_tokens_to_ids(config.short_end_token)
        self.long_invoke_id  = tokenizer.convert_tokens_to_ids(config.long_invoke_token)
        self.long_end_id     = tokenizer.convert_tokens_to_ids(config.long_end_token)

        if any(x is None or x == tokenizer.unk_token_id for x in [self.short_invoke_id, self.short_end_id, self.long_invoke_id, self.long_end_id]):
            raise ValueError("Special tokens not found in tokenizer. Make sure to call add_vismem_tokens().")

        if config.visual_feature_source not in {"projected", "last_hidden"}:
            raise ValueError(f"Unknown visual_feature_source: {config.visual_feature_source!r}")
        self.visual_pad_ids = [
            tid
            for tid in (tokenizer.convert_tokens_to_ids(tok) for tok in ("<|image_pad|>", "<|video_pad|>"))
            if tid is not None and tid != tokenizer.unk_token_id
        ]

        # The backbone stays frozen; SelectStream modules live on its device and dtype.
        ref = next(base_model.parameters())
        for module in (
            self.query_builder,
            self.short_former,
            self.long_former,
            self.lvm_short_former,
            self.lvm_long_former,
            self.graph_reasoner,
        ):
            if module is not None:
                module.to(device=ref.device, dtype=ref.dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    def _select_visual_positions(self, input_ids: torch.LongTensor) -> torch.BoolTensor:
        # Image/video pad positions hold the projected visual embeddings Phi_v(x_t).
        if self.visual_pad_ids:
            pad_mask = torch.isin(input_ids, torch.tensor(self.visual_pad_ids, device=input_ids.device))
            if pad_mask.any():
                return pad_mask
        # Fallback: the <|vision_start|> ... <|vision_end|> span.
        vs_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        if vs_id is None or ve_id is None or vs_id == self.tokenizer.unk_token_id or ve_id == self.tokenizer.unk_token_id:
            # Fallback: no visual positions
            return torch.zeros_like(input_ids, dtype=torch.bool)

        B, T = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(B):
            ids = input_ids[b].tolist()
            try:
                s = ids.index(vs_id)
                e = ids.index(ve_id)
                if e > s:
                    mask[b, s:e+1] = True
            except ValueError:
                pass
        return mask

    def _format_prompt(self, prompt: str, with_image: bool) -> str:
        if "<|im_start|>" in prompt or "<|vision_start|>" in prompt:
            return prompt  # already chat-formatted
        content: List[Dict[str, Any]] = [{"type": "image"}] if with_image else []
        content.append({"type": "text", "text": prompt})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def prepare_inputs(self, images: Optional[List[Any]], prompts: List[str]) -> Dict[str, Any]:
        """Wrap raw prompts in the backbone chat template (with image placeholders) and run the processor."""
        if images is None:
            images = [None] * len(prompts)
        if len(images) != len(prompts):
            raise ValueError(f"`prompts` length must match number of images, got {len(prompts)} vs {len(images)}.")
        texts = [self._format_prompt(p, img is not None) for p, img in zip(prompts, images)]
        present = [img for img in images if img is not None]
        if present:
            inputs = self.processor(text=texts, images=present, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def visual_feature_states(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # "projected": decoder input embeddings, where visual positions hold the output of
        # the frozen visual tower + native multimodal projector, i.e. Phi_v(x_t).
        if self.cfg.visual_feature_source == "last_hidden":
            return hidden_states[-1]
        return hidden_states[0]

    def continue_forward(
        self,
        past_key_values,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Append tokens after a cached prefix. Explicit cache positions keep Qwen-VL M-RoPE
        continuing from the prefix (via its rope deltas) instead of restarting at position 0."""
        x = input_ids if input_ids is not None else inputs_embeds
        past_len = past_key_values.get_seq_length()
        cache_position = torch.arange(past_len, past_len + x.size(1), device=x.device)
        return self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
            **kwargs,
        )

    def _build_H(self, visual_states: torch.Tensor, text_states: torch.Tensor) -> torch.Tensor:
        # Cap length to reduce compute
        if text_states.size(1) > self.cfg.max_prompt_hidden:
            text_states = text_states[:, -self.cfg.max_prompt_hidden:, :]
        return torch.cat([visual_states, text_states], dim=1)


    def _maybe_project_short_memory(self, M: torch.Tensor) -> torch.Tensor:
        proj = (
                getattr(self.base_model, "visual_projector", None)
                or getattr(self.base_model, "vision_projector", None)
                or getattr(self.base_model, "multi_modal_projector", None)
        )
        if proj is None:
            return M
        try:
            return proj(M)
        except Exception:
            return M


    def _former_forward_lora(self, X: torch.Tensor, Q: torch.Tensor, mem_len: int, adapter_name: str) -> torch.Tensor:
        # Use the underlying LLM forward on embeddings; assumes base_model supports inputs_embeds.
        peft_model = self.peft_model
        set_active_adapter(peft_model, adapter_name)
        B = X.size(0)

        if adapter_name == "short_former":
            m_init = self.m_init_short.expand(B, -1, -1).to(dtype=X.dtype, device=X.device)
        else:
            m_init = self.m_init_long.expand(B, -1, -1).to(dtype=X.dtype, device=X.device)

        inp = torch.cat([X, Q, m_init], dim=1)
        attn = torch.ones(B, inp.size(1), device=X.device, dtype=torch.long)
        out = peft_model(inputs_embeds=inp, attention_mask=attn, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states[-1]
        M = hs[:, -mem_len:, :]
        return M

    def form_memory(
        self,
        H: torch.Tensor,
        mem_type: str,
        memory_state: Optional[List[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        flush_saw: bool = True,
        chunk_attention: Optional[torch.Tensor] = None,
        chunk_time_spans: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, List[LVMMemoryState]]:
        if self.former_backend == "lvm_sft":
            former = self.lvm_short_former if mem_type == "short" else self.lvm_long_former
            if former is None:
                raise ValueError(f"LVM former for mem_type={mem_type!r} is not initialized.")
            if return_memory_state:
                return former(
                    H,
                    memory_state=memory_state,
                    return_memory_state=True,
                    step_stride=step_stride,
                    flush_saw=flush_saw,
                    chunk_attention=chunk_attention,
                    chunk_time_spans=chunk_time_spans,
                )
            return former(
                H,
                memory_state=memory_state,
                step_stride=step_stride,
                flush_saw=flush_saw,
                chunk_attention=chunk_attention,
                chunk_time_spans=chunk_time_spans,
            )

        Q = self.query_builder(H)
        if self.peft_model is None:
            if mem_type == "short":
                return self.short_former(H, Q)
            else:
                return self.long_former(H, Q)
        else:
            if mem_type == "short":
                return self._former_forward_lora(H, Q, self.cfg.short_mem_len, "short_former")
            else:
                return self._former_forward_lora(H, Q, self.cfg.long_mem_len, "long_former")

    def init_memory_state(
        self,
        batch_size: int = 1,
        mem_type: str = "short",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[List[LVMMemoryState]]:
        if self.former_backend != "lvm_sft":
            return None
        former = self.lvm_short_former if mem_type == "short" else self.lvm_long_former
        if former is None:
            return None
        return former.init_memory_state(batch_size=batch_size, device=device or self.device, dtype=dtype)

    def get_lvm_debug_stats(self, mem_type: str = "short") -> Optional[Dict[str, float]]:
        if self.former_backend != "lvm_sft":
            return None
        former = self.lvm_short_former if mem_type == "short" else self.lvm_long_former
        if former is None:
            return None
        return former.get_last_summary()

    def _gather_padded(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # states: (B,T,D), mask: (B,T)
        B, T, D = states.shape
        lens = mask.sum(dim=1)
        max_len = int(lens.max().item()) if lens.numel() else 0
        out = states.new_zeros((B, max_len, D))
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[b, : idx.numel()] = states[b, idx]
        return out

    def _gather_padded_with_mask(
        self,
        states: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        # states: (B,T,D), mask: (B,T) -> gathered states plus valid-token mask.
        B, T, D = states.shape
        lens = mask.sum(dim=1)
        max_len = int(lens.max().item()) if lens.numel() else 0
        out = states.new_zeros((B, max_len, D))
        out_mask = torch.zeros((B, max_len), device=states.device, dtype=torch.bool)
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[b, : idx.numel()] = states[b, idx]
                out_mask[b, : idx.numel()] = True
        return out, out_mask

    def _masked_mean(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # states: [B,T,D], mask: [B,T] -> [B,D]
        B, T, D = states.shape
        out = states.new_zeros(B, D)
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                out[b] = states[b, idx].mean(dim=0)
            else:
                out[b] = states[b].mean(dim=0)
        return out

    def _extract_visual_states(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor) -> torch.Tensor:
        visual_mask = self._select_visual_positions(input_ids)
        if visual_mask.any():
            return self._gather_padded(hidden_states, visual_mask)
        return hidden_states

    def _extract_text_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        visual_mask = self._select_visual_positions(input_ids)
        text_mask = ~visual_mask
        if attention_mask is not None:
            text_mask = text_mask & attention_mask.bool()
        text_states, text_valid = self._gather_padded_with_mask(hidden_states, text_mask)
        if text_states.size(1) > self.cfg.max_prompt_hidden:
            text_states = text_states[:, -self.cfg.max_prompt_hidden :, :]
            text_valid = text_valid[:, -self.cfg.max_prompt_hidden :]
        return text_states, text_valid

    def encode_query(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_states, text_valid = self._extract_text_states(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if text_states.size(1) == 0:
            return hidden_states.mean(dim=1)
        return self.query_builder(
            text_states,
            H_key_padding_mask=~text_valid,
        )

    def needs_saw_attention(self, mem_type: str = "short") -> bool:
        if self.former_backend != "lvm_sft":
            return False
        former = self.lvm_short_former if mem_type == "short" else self.lvm_long_former
        return bool(former is not None and former.use_saw)

    def _extract_chunk_attention(
        self,
        attentions: Optional[Any],
        input_ids: torch.LongTensor,
        groups: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if attentions is None:
            return None
        if isinstance(attentions, tuple) or isinstance(attentions, list):
            if len(attentions) == 0:
                return None
            attn_last = attentions[-1]
        else:
            attn_last = attentions

        if attn_last is None or attn_last.dim() != 4:
            return None

        bsz, _, _, _ = attn_last.shape
        visual_mask = self._select_visual_positions(input_ids)
        text_mask = ~visual_mask
        groups = groups or self.cfg.lvm_saw_attn_groups
        out: List[torch.Tensor] = []
        for b in range(bsz):
            vis_idx = visual_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if vis_idx.numel() == 0:
                out.append(attn_last.new_full((groups,), 1.0 / float(groups)))
                continue

            query_idx = text_mask[b].nonzero(as_tuple=False).squeeze(-1)
            attn_b = attn_last[b]  # [H, T, T]
            if query_idx.numel() > 0:
                visual_attn = attn_b[:, query_idx][:, :, vis_idx].mean(dim=(0, 1))
            else:
                visual_attn = attn_b[:, :, vis_idx].mean(dim=(0, 1))

            visual_attn = visual_attn.clamp(min=0)
            if float(visual_attn.sum().item()) <= 0:
                visual_attn = torch.full_like(visual_attn, 1.0 / float(max(1, visual_attn.numel())))
            else:
                visual_attn = visual_attn / visual_attn.sum()

            pooled = F.adaptive_avg_pool1d(visual_attn.view(1, 1, -1), groups).view(-1)
            if float(pooled.sum().item()) <= 0:
                pooled = pooled.new_full((groups,), 1.0 / float(groups))
            else:
                pooled = pooled / pooled.sum()
            out.append(pooled)
        return torch.stack(out, dim=0)

    def _update_memory_impl(
        self,
        images: Optional[List[Any]],
        prompts: Optional[List[str]] = None,
        mem_type: str = "short",
        memory_state: Optional[List[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        flush_saw: bool = False,
        chunk_time_spans: Optional[torch.Tensor] = None,
    ):
        """
        Shared path for streaming memory updates.
        """
        if images is None:
            raise ValueError("`images` must be provided when updating memory.")
        if prompts is None:
            prompts = [""] * len(images)
        inputs = self.prepare_inputs(images, prompts)

        need_attn = self.needs_saw_attention(mem_type=mem_type)
        with torch.no_grad():
            out = self.base_model(**inputs, output_hidden_states=True, output_attentions=need_attn)
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("Processor did not return input_ids; check your Qwen2.5-VL processor.")

        visual_states = self._extract_visual_states(self.visual_feature_states(out.hidden_states), input_ids)
        chunk_attention = self._extract_chunk_attention(
            getattr(out, "attentions", None),
            input_ids,
        ) if need_attn else None
        if return_memory_state:
            _, next_memory_state = self.form_memory(
                visual_states,
                mem_type=mem_type,
                memory_state=memory_state,
                return_memory_state=True,
                step_stride=step_stride,
                flush_saw=flush_saw,
                chunk_attention=chunk_attention,
                chunk_time_spans=chunk_time_spans,
            )
            return next_memory_state
        return self.form_memory(
            visual_states,
            mem_type=mem_type,
            memory_state=memory_state,
            step_stride=step_stride,
            flush_saw=flush_saw,
            chunk_attention=chunk_attention,
            chunk_time_spans=chunk_time_spans,
        )

    @torch.no_grad()
    def update_memory(
        self,
        images: Optional[List[Any]],
        prompts: Optional[List[str]] = None,
        mem_type: str = "short",
        memory_state: Optional[List[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        flush_saw: bool = False,
        chunk_time_spans: Optional[torch.Tensor] = None,
    ):
        """
        Inference-time memory update without tracking gradients.
        """
        return self._update_memory_impl(
            images=images,
            prompts=prompts,
            mem_type=mem_type,
            memory_state=memory_state,
            return_memory_state=return_memory_state,
            step_stride=step_stride,
            flush_saw=flush_saw,
            chunk_time_spans=chunk_time_spans,
        )

    def update_memory_with_grad(
        self,
        images: Optional[List[Any]],
        prompts: Optional[List[str]] = None,
        mem_type: str = "short",
        memory_state: Optional[List[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        flush_saw: bool = False,
        chunk_time_spans: Optional[torch.Tensor] = None,
    ):
        """
        Training-time memory update that keeps the LVM write path differentiable.
        """
        return self._update_memory_impl(
            images=images,
            prompts=prompts,
            mem_type=mem_type,
            memory_state=memory_state,
            return_memory_state=return_memory_state,
            step_stride=step_stride,
            flush_saw=flush_saw,
            chunk_time_spans=chunk_time_spans,
        )

    def build_method_evidence(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_state: Optional[List[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        flush_saw: bool = True,
        chunk_attention: Optional[torch.Tensor] = None,
        chunk_time_spans: Optional[torch.Tensor] = None,
        visual_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.BoolTensor, List[Dict[str, torch.Tensor]]] | tuple[torch.Tensor, torch.BoolTensor, List[Dict[str, torch.Tensor]], List[LVMMemoryState]]:
        """
        Build latent evidence tokens:
        projected visual embeddings -> SAW/LVM write -> query-conditioned subgraph retrieval + GAR.

        `hidden_states` are the prompt-side states used by the query encoder; `visual_hidden_states`
        (default: `hidden_states`) are the states written to memory. Returns evidence tokens
        [B, M', D] (left-padded across the batch), their validity mask [B, M'] and per-sample aux.
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"`hidden_states` must be [B,T,D], got {tuple(hidden_states.shape)}")
        if input_ids.dim() != 2:
            raise ValueError(f"`input_ids` must be [B,T], got {tuple(input_ids.shape)}")

        if visual_hidden_states is None:
            visual_hidden_states = hidden_states
        visual_states = self._extract_visual_states(visual_hidden_states, input_ids)
        query_states = self.encode_query(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Run LVM write/read first; this also stores per-batch slot states in lvm_short_former.
        next_memory_state = None
        if return_memory_state:
            memory_tokens, next_memory_state = self.form_memory(
                visual_states,
                mem_type="short",
                memory_state=memory_state,
                return_memory_state=True,
                step_stride=step_stride,
                flush_saw=flush_saw,
                chunk_attention=chunk_attention,
                chunk_time_spans=chunk_time_spans,
            )
        else:
            memory_tokens = self.form_memory(
                visual_states,
                mem_type="short",
                memory_state=memory_state,
                step_stride=step_stride,
                flush_saw=flush_saw,
                chunk_attention=chunk_attention,
                chunk_time_spans=chunk_time_spans,
            )

        if self.former_backend != "lvm_sft" or self.lvm_short_former is None:
            evidence_mask = torch.ones(memory_tokens.shape[:2], device=memory_tokens.device, dtype=torch.bool)
            if return_memory_state:
                return memory_tokens, evidence_mask, [], next_memory_state or []
            return memory_tokens, evidence_mask, []

        aux_list: List[Dict[str, torch.Tensor]] = []
        evidence_list: List[torch.Tensor] = []
        for b in range(hidden_states.size(0)):
            state = self.lvm_short_former.get_last_state(b)
            if state is None:
                evidence_list.append(memory_tokens[b])
                aux_list.append({})
                continue

            active_slots = state["active_slots"]
            if active_slots.numel() == 0 or self.graph_reasoner is None:
                # No GAR: expose the legacy readout slots (none if memory is still empty).
                evidence_list.append(state["selected_slots"])
                aux_list.append(
                    {
                        "evidence_indices": state["selected_indices"],
                        "node_indices": state["active_indices"],
                        "node_times": state["active_times"],
                        "node_time_spans": torch.stack([state["active_time_starts"], state["active_time_ends"]], dim=-1)
                        if state["active_time_starts"].numel() > 0
                        else torch.empty(0, 2, device=hidden_states.device, dtype=hidden_states.dtype),
                        "node_merge_count": state.get("active_merge_count", torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "graph_edge_index": state.get("graph_edge_index", torch.empty(2, 0, device=hidden_states.device, dtype=torch.long)),
                        "graph_edge_type": state.get("graph_edge_type", torch.empty(0, device=hidden_states.device, dtype=torch.long)),
                        "graph_edge_weight": state.get("graph_edge_weight", torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "merge_count": state.get("merge_count", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "merge_penalty_similarity": state.get("merge_penalty_similarity", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "merge_penalty_surprise": state.get("merge_penalty_surprise", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "merge_penalty_access": state.get("merge_penalty_access", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "merge_penalty_recency": state.get("merge_penalty_recency", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "merge_penalty_total": state.get("merge_penalty_total", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)),
                        "last_merge_pair": state.get("last_merge_pair", torch.full((2,), -1, device=hidden_states.device, dtype=torch.long)),
                        "subgraph_size": torch.tensor(float(state["selected_indices"].numel()), device=hidden_states.device),
                        "sparsity_loss": torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
                    }
                )
                if next_memory_state:
                    LVMMemoryFormer.mark_read(next_memory_state[b], state["selected_indices"])
                continue

            reason_out = self.graph_reasoner(
                node_feats=active_slots,
                node_time_start=state["active_time_starts"],
                node_time_end=state["active_time_ends"],
                node_surprise=state["active_surprise"],
                node_indices=state["active_indices"],
                query=query_states[b],
                node_merge_count=state.get("active_merge_count", torch.zeros_like(state["active_surprise"])),
                edge_index=state.get("graph_edge_index", None),
                edge_type=state.get("graph_edge_type", None),
                edge_weight=state.get("graph_edge_weight", None),
            )
            reason_out["merge_count"] = state.get("merge_count", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype))
            reason_out["merge_penalty_similarity"] = state.get("merge_penalty_similarity", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype))
            reason_out["merge_penalty_surprise"] = state.get("merge_penalty_surprise", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype))
            reason_out["merge_penalty_access"] = state.get("merge_penalty_access", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype))
            reason_out["merge_penalty_recency"] = state.get("merge_penalty_recency", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype))
            reason_out["merge_penalty_total"] = state.get("merge_penalty_total", torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype))
            reason_out["last_merge_pair"] = state.get("last_merge_pair", torch.full((2,), -1, device=hidden_states.device, dtype=torch.long))
            evidence_list.append(reason_out["evidence_tokens"])
            aux_list.append(reason_out)
            if next_memory_state:
                # Read counts track how often a node is exposed as query evidence.
                LVMMemoryFormer.mark_read(next_memory_state[b], reason_out["evidence_indices"])

        # Only the retrieved evidence is injected; shorter rows are left-padded and masked out.
        max_len = max((int(e.size(0)) for e in evidence_list), default=0)
        evidence_tokens = memory_tokens.new_zeros(len(evidence_list), max_len, self.hidden_size)
        evidence_mask = torch.zeros(len(evidence_list), max_len, device=memory_tokens.device, dtype=torch.bool)
        for b, ev in enumerate(evidence_list):
            n = int(ev.size(0))
            if n > 0:
                evidence_tokens[b, max_len - n :] = ev.to(dtype=evidence_tokens.dtype)
                evidence_mask[b, max_len - n :] = True
        if return_memory_state:
            return evidence_tokens, evidence_mask, aux_list, next_memory_state or []
        return evidence_tokens, evidence_mask, aux_list

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _aux_to_python(aux_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for aux in aux_list:
            item: Dict[str, Any] = {}
            for key, value in aux.items():
                if isinstance(value, torch.Tensor):
                    t = value.detach().cpu()
                    item[key] = t.tolist() if t.dim() > 0 else t.item()
                else:
                    item[key] = value
            out.append(item)
        return out

    @torch.no_grad()
    def generate_method(
        self,
        images: Optional[List[Any]],
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_token_ids: bool = False,
        skip_special_tokens: bool = True,
        return_aux: bool = False,
        memory_state: Optional[List[LVMMemoryState] | LVMMemoryState] = None,
        return_memory_state: bool = False,
        step_stride: float = 1.0,
        chunk_time_spans: Optional[torch.Tensor] = None,
    ):
        """
        SelectStream inference (Eq. 11): X_in = [X_prompt; X_cur; e_1..e_M] -> decode answer.
        The current observation stays directly visible; history enters only as evidence tokens.
        """
        inputs = self.prepare_inputs(images, prompts)

        need_attn = self.needs_saw_attention(mem_type="short")
        out = self.base_model(**inputs, use_cache=True, output_hidden_states=True, output_attentions=need_attn)
        past = out.past_key_values
        hidden_last = out.hidden_states[-1]

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("Processor did not return input_ids; check your Qwen2.5-VL processor.")
        chunk_attention = self._extract_chunk_attention(
            getattr(out, "attentions", None),
            input_ids,
        ) if need_attn else None

        next_memory_state = None
        evidence_out = self.build_method_evidence(
            hidden_last,
            input_ids,
            attention_mask=inputs.get("attention_mask", None),
            memory_state=memory_state,
            return_memory_state=return_memory_state,
            step_stride=step_stride,
            chunk_attention=chunk_attention,
            chunk_time_spans=chunk_time_spans,
            visual_hidden_states=self.visual_feature_states(out.hidden_states),
        )
        if return_memory_state:
            evidence_tokens, evidence_mask, aux_list, next_memory_state = evidence_out
        else:
            evidence_tokens, evidence_mask, aux_list = evidence_out

        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        cur_logits = out.logits[:, -1, :]
        if evidence_tokens.size(1) > 0:
            attn_mask = torch.cat([attn_mask, evidence_mask.to(dtype=attn_mask.dtype)], dim=1)
            emb_dtype = self.base_model.get_input_embeddings().weight.dtype
            out = self.continue_forward(
                past,
                inputs_embeds=evidence_tokens.to(dtype=emb_dtype),
                attention_mask=attn_mask,
            )
            past = out.past_key_values
            has_evidence = evidence_mask.any(dim=1, keepdim=True)
            cur_logits = torch.where(has_evidence, out.logits[:, -1, :], cur_logits)

        generated: List[torch.Tensor] = []
        for _ in range(max_new_tokens):
            next_id = self._sample_next_token(cur_logits, temperature=temperature, top_p=top_p)
            generated.append(next_id)
            if (next_id == self.tokenizer.eos_token_id).all():
                break
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones(attn_mask.size(0), 1)], dim=1)
            out = self.continue_forward(past, input_ids=next_id.unsqueeze(-1), attention_mask=attn_mask)
            past = out.past_key_values
            cur_logits = out.logits[:, -1, :]

        if generated:
            gen_ids = torch.stack(generated, dim=1)
        else:
            gen_ids = torch.empty((input_ids.size(0), 0), device=self.device, dtype=torch.long)

        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_special_tokens)
        py_aux = self._aux_to_python(aux_list) if return_aux else None
        if return_token_ids and return_aux and return_memory_state:
            return texts, gen_ids, py_aux, next_memory_state
        if return_token_ids and return_aux:
            return texts, gen_ids, py_aux
        if return_token_ids and return_memory_state:
            return texts, gen_ids, next_memory_state
        if return_token_ids:
            return texts, gen_ids
        if return_aux and return_memory_state:
            return texts, py_aux, next_memory_state
        if return_aux:
            return texts, py_aux
        if return_memory_state:
            return texts, next_memory_state
        return texts

    @torch.no_grad()
    def generate(
        self,
        images: Optional[List[Any]],
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_vismem: bool = True,
        return_token_ids: bool = False,
        skip_special_tokens: bool = True,
        reverse_mem_type: bool = False,
        ):
        # batch size 1 recommended; we keep batch support
        inputs = self.prepare_inputs(images, prompts)

        # Initial forward
        out = self.base_model(**inputs, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        hidden_last = out.hidden_states[-1]  # (B, T, D)

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("Processor did not return input_ids; check your Qwen2.5-VL processor.")
        B, T = input_ids.shape

        visual_mask = self._select_visual_positions(input_ids)

        if visual_mask.any():
            visual_states = self._gather_padded(hidden_last, visual_mask)
        else:
            visual_states = torch.zeros(B, 0, self.hidden_size, device=self.device, dtype=hidden_last.dtype)

        seg_hiddens: List[torch.Tensor] = []  # each (B,1,D)

        generated = []

        # decoding loop
        cur_logits = logits
        for step in range(max_new_tokens):
            next_id = self._sample_next_token(cur_logits, temperature=temperature, top_p=top_p)
            generated.append(next_id)
            seg_hiddens.append(hidden_last[:, -1:, :])

            # Check invocation
            if enable_vismem and (((next_id == self.short_invoke_id).any()) or ((next_id == self.long_invoke_id).any())):
                # Feed invocation token
                out = self.continue_forward(past, input_ids=next_id.unsqueeze(-1), output_hidden_states=True)
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]  # (B,1,D)
                seg_hiddens.append(hidden_last)

                token_type = "short" if (next_id == self.short_invoke_id).any() else "long"
                mem_type = (
                    "long" if token_type == "short" else "short") if reverse_mem_type else token_type

                end_id = self.short_end_id if mem_type == "short" else self.long_end_id

                # Build H
                text_states = torch.cat(seg_hiddens, dim=1)  # (B, z, D)
                H = self._build_H(visual_states, text_states)

                M = self.form_memory(H, mem_type)  # (B, N, D)
                if mem_type == "short":
                    M = self._maybe_project_short_memory(M)
                # Insert memory tokens
                out = self.continue_forward(past, inputs_embeds=M, output_hidden_states=True)
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]  # (B,N,D)

                # end
                end_tensor = torch.full((B,), end_id, device=self.device, dtype=torch.long)
                generated.append(end_tensor)
                out = self.continue_forward(past, input_ids=end_tensor.unsqueeze(-1), output_hidden_states=True)
                past = out.past_key_values
                hidden_last = out.hidden_states[-1]
                seg_hiddens = []  # reset

                cur_logits = out.logits[:, -1, :]
                continue

            # Normal step: feed token to model
            out = self.continue_forward(past, input_ids=next_id.unsqueeze(-1), output_hidden_states=True)
            past = out.past_key_values
            hidden_last = out.hidden_states[-1]  # (B,1,D)
            cur_logits = out.logits[:, -1, :]

            # stop token
            if (next_id == self.tokenizer.eos_token_id).all():
                break

        gen_ids = torch.stack(generated, dim=1)  # (B, Lg)
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_special_tokens)
        if return_token_ids:
            return texts, gen_ids
        return texts
