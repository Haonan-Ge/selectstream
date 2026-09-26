from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EVIDENCE_ID_KEYS: Sequence[str] = (
    "evidence_ids",
    "evidence_slots",
    "evidence_nodes",
    "evidence_indices",
)
EVIDENCE_TIME_KEYS: Sequence[str] = (
    "evidence_timestamps",
    "evidence_times",
    "timestamps",
    "time_spans",
    "evidence_spans",
)


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _extract_evidence_ids(meta: Optional[Dict[str, Any]]) -> List[int]:
    if not isinstance(meta, dict):
        return []
    for key in EVIDENCE_ID_KEYS:
        if key not in meta or meta[key] is None:
            continue
        value = meta[key]
        out: List[int] = []
        if isinstance(value, list):
            for x in value:
                if isinstance(x, dict):
                    cand = x.get("id", x.get("node_id", x.get("index", None)))
                    iv = _to_int(cand)
                else:
                    iv = _to_int(x)
                if iv is not None:
                    out.append(iv)
        elif isinstance(value, dict):
            cand = value.get("id", value.get("node_id", value.get("index", None)))
            iv = _to_int(cand)
            if iv is not None:
                out.append(iv)
        else:
            iv = _to_int(value)
            if iv is not None:
                out.append(iv)
        if out:
            return out
    return []


def _normalize_time_item(item: Any) -> List[Tuple[float, float]]:
    # Accept scalar, [t], [s, e], [[s,e], ...], or {"start":..., "end":...}.
    if isinstance(item, dict):
        s = _to_float(item.get("start", item.get("s", item.get("begin", item.get("t", None)))))
        e = _to_float(item.get("end", item.get("e", item.get("finish", item.get("t", None)))))
        if s is None and e is None:
            return []
        if s is None:
            s = e
        if e is None:
            e = s
        lo, hi = (s, e) if s <= e else (e, s)
        return [(lo, hi)]
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return []
        if len(item) == 2 and _to_float(item[0]) is not None and _to_float(item[1]) is not None:
            s = float(item[0])
            e = float(item[1])
            lo, hi = (s, e) if s <= e else (e, s)
            return [(lo, hi)]
        out: List[Tuple[float, float]] = []
        for x in item:
            out.extend(_normalize_time_item(x))
        return out
    v = _to_float(item)
    if v is None:
        return []
    return [(v, v)]


def _extract_evidence_time_intervals(meta: Optional[Dict[str, Any]]) -> List[Tuple[float, float]]:
    if not isinstance(meta, dict):
        return []

    for key in EVIDENCE_TIME_KEYS:
        if key in meta and meta[key] is not None:
            out = _normalize_time_item(meta[key])
            if out:
                return out

    # Fallback: paired start/end fields.
    if "evidence_start" in meta or "evidence_end" in meta:
        s = meta.get("evidence_start", None)
        e = meta.get("evidence_end", None)
        if s is not None or e is not None:
            return _normalize_time_item([s, e])

    if "timestamp" in meta and meta["timestamp"] is not None:
        return _normalize_time_item(meta["timestamp"])

    return []


def _build_id_target(node_indices: torch.Tensor, gt_ids: List[int]) -> Optional[torch.Tensor]:
    if node_indices.numel() == 0 or len(gt_ids) == 0:
        return None
    gt = torch.tensor(gt_ids, device=node_indices.device, dtype=node_indices.dtype)
    return torch.isin(node_indices, gt).to(dtype=torch.float32)


def _build_time_target(
    node_times: torch.Tensor,
    intervals: List[Tuple[float, float]],
    tolerance: float,
) -> Optional[torch.Tensor]:
    if node_times.numel() == 0 or len(intervals) == 0:
        return None
    hit: torch.Tensor
    tol = max(0.0, float(tolerance))
    if node_times.dim() == 2 and node_times.size(-1) == 2:
        start = node_times[:, 0].float()
        end = node_times[:, 1].float()
        hit = torch.zeros_like(start, dtype=torch.bool)
        for s, e in intervals:
            lo = min(float(s), float(e)) - tol
            hi = max(float(s), float(e)) + tol
            overlap = torch.minimum(end, end.new_full(end.shape, hi)) - torch.maximum(start, start.new_full(start.shape, lo))
            hit = hit | (overlap >= 0)
        return hit.to(dtype=torch.float32)

    t = node_times.float()
    hit = torch.zeros_like(t, dtype=torch.bool)
    for s, e in intervals:
        lo = min(float(s), float(e)) - tol
        hi = max(float(s), float(e)) + tol
        hit = hit | ((t >= lo) & (t <= hi))
    return hit.to(dtype=torch.float32)


def _multi_positive_retrieval_loss(node_scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.to(device=node_scores.device, dtype=torch.bool)
    if node_scores.numel() == 0 or not bool(positive.any().item()):
        return node_scores.new_zeros(())
    log_probs = F.log_softmax(node_scores, dim=0)
    return -torch.logsumexp(log_probs[positive], dim=0)


def _hit_at_k(ids: torch.Tensor, scores: torch.Tensor, gt_ids: List[int], topk: int) -> float:
    if ids.numel() == 0 or scores.numel() == 0 or len(gt_ids) == 0:
        return 0.0
    k = min(max(1, int(topk)), scores.numel())
    picked = ids[torch.topk(scores, k=k, largest=True).indices]
    gt = torch.tensor(gt_ids, device=ids.device, dtype=ids.dtype)
    return float(torch.isin(picked, gt).any().item())


def _time_hit_at_k(
    node_times: torch.Tensor,
    scores: torch.Tensor,
    intervals: List[Tuple[float, float]],
    topk: int,
    tolerance: float,
) -> float:
    if node_times.numel() == 0 or scores.numel() == 0 or len(intervals) == 0:
        return 0.0
    k = min(max(1, int(topk)), scores.numel())
    picked_t = node_times[torch.topk(scores, k=k, largest=True).indices].float()
    tol = max(0.0, float(tolerance))
    if picked_t.dim() == 2 and picked_t.size(-1) == 2:
        picked_start = picked_t[:, 0]
        picked_end = picked_t[:, 1]
        hit = torch.zeros_like(picked_start, dtype=torch.bool)
        for s, e in intervals:
            lo = min(float(s), float(e)) - tol
            hi = max(float(s), float(e)) + tol
            overlap = torch.minimum(picked_end, picked_end.new_full(picked_end.shape, hi)) - torch.maximum(
                picked_start,
                picked_start.new_full(picked_start.shape, lo),
            )
            hit = hit | (overlap >= 0)
        return float(hit.any().item())
    hit = torch.zeros_like(picked_t, dtype=torch.bool)
    for s, e in intervals:
        lo = min(float(s), float(e)) - tol
        hi = max(float(s), float(e)) + tol
        hit = hit | ((picked_t >= lo) & (picked_t <= hi))
    return float(hit.any().item())


def _temporal_overlap(
    spans: torch.Tensor,
    intervals: List[Tuple[float, float]],
    frame_duration: float,
) -> float:
    """
    Best fraction |I_i ∩ g| / |g| over retrieved node spans I_i and annotated intervals g (Eqs. 56-57).
    Each sampled frame covers half a frame period on either side of its timestamp.
    """
    if spans.numel() == 0 or len(intervals) == 0:
        return 0.0
    half = 0.5 * max(float(frame_duration), 1e-6)
    best = 0.0
    for s, e in spans.reshape(-1, 2).tolist():
        node_lo, node_hi = min(s, e) - half, max(s, e) + half
        for gs, ge in intervals:
            lo, hi = min(gs, ge), max(gs, ge)
            if hi - lo < 2.0 * half:  # point-like annotation: one frame period
                lo, hi = lo - half, hi + half
            inter = max(0.0, min(node_hi, hi) - max(node_lo, lo))
            best = max(best, inter / (hi - lo))
    return min(best, 1.0)


def method_sft_loss(
    base_model,
    selectstream_model,
    inputs: Dict[str, Any],
    target_text: str,
    meta: Optional[Dict[str, Any]],
    memory_state: Optional[Any] = None,
    step_stride: float = 1.0,
    chunk_time_spans: Optional[torch.Tensor] = None,
    beta_ret: float = 0.0,
    gamma_spar: float = 0.0,
    ret_id_weight: float = 1.0,
    ret_time_weight: float = 1.0,
    time_tolerance: float = 0.0,
    evidence_topk: int = 8,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    SelectStream objective (Eq. 14):
    L = L_ans + beta * L_ret + gamma * L_spar
    """
    tokenizer = selectstream_model.tokenizer
    device = selectstream_model.device

    tgt_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)

    # Prompt pass over [X_prompt; X_cur] with the current observation visible to the frozen
    # backbone. Its KV cache is reused so the answer pass matches inference exactly.
    need_attn = selectstream_model.needs_saw_attention(mem_type="short")
    with torch.no_grad():
        out_prompt = base_model(**inputs, use_cache=True, output_hidden_states=True, output_attentions=need_attn)
    input_ids = inputs["input_ids"]
    chunk_attention = None
    if need_attn:
        chunk_attention = selectstream_model._extract_chunk_attention(
            getattr(out_prompt, "attentions", None),
            input_ids,
        )

    evidence_tokens, evidence_mask, aux_list = selectstream_model.build_method_evidence(
        out_prompt.hidden_states[-1],
        input_ids,
        attention_mask=inputs.get("attention_mask", None),
        memory_state=memory_state,
        step_stride=step_stride,
        chunk_attention=chunk_attention,
        chunk_time_spans=chunk_time_spans,
        visual_hidden_states=selectstream_model.visual_feature_states(out_prompt.hidden_states),
    )

    # Answer pass: [e_1..e_M; y] on top of the cached prompt (Eq. 11).
    emb_layer = base_model.get_input_embeddings()
    tgt_emb = emb_layer(tgt_ids)
    ans_embeds = torch.cat([evidence_tokens.to(dtype=tgt_emb.dtype), tgt_emb], dim=1)
    prompt_mask = inputs.get("attention_mask", None)
    if prompt_mask is None:
        prompt_mask = torch.ones_like(input_ids)
    attn = torch.cat(
        [
            prompt_mask,
            evidence_mask.to(dtype=prompt_mask.dtype),
            torch.ones(tgt_ids.shape, device=device, dtype=prompt_mask.dtype),
        ],
        dim=1,
    )
    out = selectstream_model.continue_forward(
        out_prompt.past_key_values,
        inputs_embeds=ans_embeds,
        attention_mask=attn,
    )
    # The logit predicting y_j sits one position earlier; y_0 is predicted from the last
    # evidence token, or from the last prompt token when no evidence is available.
    all_logits = torch.cat([out_prompt.logits[:, -1:, :].to(dtype=out.logits.dtype), out.logits], dim=1)
    num_ev = evidence_tokens.size(1)
    logits = all_logits[:, num_ev : num_ev + tgt_ids.size(1), :]
    loss_ans = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt_ids.reshape(-1))

    gt_ids = _extract_evidence_ids(meta)
    gt_intervals = _extract_evidence_time_intervals(meta)

    loss_ret_id = torch.tensor(0.0, device=device, dtype=loss_ans.dtype)
    loss_ret_time = torch.tensor(0.0, device=device, dtype=loss_ans.dtype)
    hit_id = torch.tensor(0.0, device=device, dtype=loss_ans.dtype)
    hit_time = torch.tensor(0.0, device=device, dtype=loss_ans.dtype)
    loss_ret = torch.tensor(0.0, device=device, dtype=loss_ans.dtype)
    loss_spar = torch.tensor(0.0, device=device, dtype=loss_ans.dtype)

    if aux_list:
        aux = aux_list[0]
        node_scores = aux.get("node_scores", torch.empty(0, device=device, dtype=loss_ans.dtype))
        node_indices = aux.get("node_indices", torch.empty(0, device=device, dtype=torch.long))
        node_times = aux.get("node_time_spans", aux.get("node_times", torch.empty(0, device=device, dtype=loss_ans.dtype)))

        id_target = _build_id_target(node_indices, gt_ids)
        time_target = _build_time_target(node_times, gt_intervals, time_tolerance)

        has_id = id_target is not None and node_scores.numel() == id_target.numel()
        has_time = time_target is not None and node_scores.numel() == time_target.numel()

        if has_id:
            loss_ret_id = _multi_positive_retrieval_loss(node_scores, id_target)
            hit_id = torch.tensor(_hit_at_k(node_indices, node_scores, gt_ids, evidence_topk), device=device, dtype=loss_ans.dtype)
        if has_time:
            loss_ret_time = _multi_positive_retrieval_loss(node_scores, time_target)
            hit_time = torch.tensor(
                _time_hit_at_k(node_times, node_scores, gt_intervals, evidence_topk, time_tolerance),
                device=device,
                dtype=loss_ans.dtype,
            )

        denom = 0.0
        if has_id and ret_id_weight > 0:
            loss_ret = loss_ret + ret_id_weight * loss_ret_id
            denom += ret_id_weight
        if has_time and ret_time_weight > 0:
            loss_ret = loss_ret + ret_time_weight * loss_ret_time
            denom += ret_time_weight
        if denom > 0:
            loss_ret = loss_ret / denom

        spar = aux.get("sparsity_loss", None)
        if isinstance(spar, torch.Tensor):
            loss_spar = spar.to(device=device, dtype=loss_ans.dtype)

    total = loss_ans + beta_ret * loss_ret + gamma_spar * loss_spar
    stats = {
        "loss_total": total.detach(),
        "loss_ans": loss_ans.detach(),
        "loss_ret": loss_ret.detach(),
        "loss_ret_id": loss_ret_id.detach(),
        "loss_ret_time": loss_ret_time.detach(),
        "loss_spar": loss_spar.detach(),
        "hit_id_at_k": hit_id.detach(),
        "hit_time_at_k": hit_time.detach(),
    }
    if aux_list and isinstance(aux_list[0], dict) and "subgraph_size" in aux_list[0]:
        sg = aux_list[0]["subgraph_size"]
        if isinstance(sg, torch.Tensor):
            stats["subgraph_size"] = sg.detach()
    if aux_list and isinstance(aux_list[0], dict):
        spar_sub = aux_list[0].get("sparsity_subgraph_loss", None)
        if isinstance(spar_sub, torch.Tensor):
            stats["loss_spar_subgraph"] = spar_sub.detach()
        spar_red = aux_list[0].get("sparsity_redundancy_loss", None)
        if isinstance(spar_red, torch.Tensor):
            stats["loss_spar_redundancy"] = spar_red.detach()
    return total, stats
