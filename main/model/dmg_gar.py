from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphState:
    node_feats: torch.Tensor
    node_time_start: torch.Tensor
    node_time_end: torch.Tensor
    node_surprise: torch.Tensor
    node_merge_count: torch.Tensor
    edge_index: torch.Tensor  # [2, E]
    edge_type: torch.Tensor   # [E], 0 temporal, 1 similarity
    edge_weight: torch.Tensor # [E]


class RelationalGAR(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.q_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.o_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.edge_type_bias = nn.Parameter(torch.zeros(2))
        # tau_t = softplus(tau_hat_t) + eps (Eq. 35), initialized so that tau_t = 1.
        self.dt_scale_raw = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Older checkpoints stored tau_t directly as `dt_scale`; convert it to the softplus parameterization.
        legacy_key = prefix + "dt_scale"
        if legacy_key in state_dict and prefix + "dt_scale_raw" not in state_dict:
            tau = state_dict.pop(legacy_key).abs().clamp(min=1e-3)
            state_dict[prefix + "dt_scale_raw"] = torch.log(torch.expm1(tau))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, graph: GraphState) -> torch.Tensor:
        x = graph.node_feats
        if x.size(0) <= 1 or graph.edge_index.numel() == 0:
            return x

        src, dst = graph.edge_index[0], graph.edge_index[1]
        s_src = graph.node_time_start[src]
        e_src = graph.node_time_end[src]
        s_dst = graph.node_time_start[dst]
        e_dst = graph.node_time_end[dst]
        dt_gap = torch.maximum(torch.maximum(s_src - e_dst, s_dst - e_src), torch.zeros_like(s_src))
        tau_t = F.softplus(self.dt_scale_raw) + 1e-6
        dt_bias = -dt_gap / tau_t
        w_bias = graph.edge_weight
        type_bias = self.edge_type_bias[graph.edge_type]

        for l in range(self.num_layers):
            q = self.q_proj[l](x)
            k = self.k_proj[l](x)
            v = self.v_proj[l](x)

            score = (q[src] * k[dst]).sum(dim=-1) / (self.hidden_size ** 0.5)
            score = score + type_bias + dt_bias + w_bias

            # Softmax over neighbors of each src node.
            attn = torch.zeros_like(score)
            for i in range(x.size(0)):
                idx = (src == i).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() > 0:
                    attn[idx] = torch.softmax(score[idx], dim=0)

            msg = torch.zeros_like(x)
            msg.index_add_(0, src, attn.unsqueeze(-1) * v[dst])
            x = x + self.o_proj[l](msg)
        return x


class DynamicMemoryGraphReasoner(nn.Module):
    """
    DMG + GAR:
    1) Build temporal/similarity graph over memory slots.
    2) Retrieve subgraph by query + surprise boost.
    3) Apply relational graph attention.
    """

    def __init__(
        self,
        hidden_size: int,
        graph_budget: int = 64,
        seed_topk: int = 16,
        num_hops: int = 2,
        evidence_tokens: int = 8,
        eta_surprise_boost: float = 0.2,
        span_penalty_weight: float = 0.05,
        merge_penalty_weight: float = 0.05,
        route_edge_weight: float = 0.1,
        route_temperature: float = 1.0,
        route_threshold: float = 0.0,
        sim_topk: int = 4,
        temporal_weight_c: float = 1.0,
        gar_layers: int = 2,
        spar_subgraph_weight: float = 1.0,
        spar_redundancy_weight: float = 1.0,
        redundancy_margin: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.graph_budget = graph_budget
        self.seed_topk = seed_topk
        self.num_hops = num_hops
        self.evidence_tokens = evidence_tokens
        self.eta_surprise_boost = eta_surprise_boost
        self.span_penalty_weight = float(span_penalty_weight)
        self.merge_penalty_weight = float(merge_penalty_weight)
        self.route_edge_weight = float(route_edge_weight)
        self.route_temperature = float(route_temperature)
        self.route_threshold = float(route_threshold)
        self.sim_topk = sim_topk
        self.temporal_weight_c = temporal_weight_c
        self.spar_subgraph_weight = float(spar_subgraph_weight)
        self.spar_redundancy_weight = float(spar_redundancy_weight)
        self.redundancy_margin = float(redundancy_margin)
        self.gar = RelationalGAR(hidden_size=hidden_size, num_layers=gar_layers)
        self.evidence_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.evidence_ln = nn.LayerNorm(hidden_size)
        nn.init.eye_(self.evidence_proj.weight)

    @staticmethod
    def _normalize_stat(x: torch.Tensor) -> torch.Tensor:
        # Divide by the maximum over the active memory, with eps for stability.
        if x.numel() == 0:
            return x
        x = x.float().clamp_min(0)
        return x / (x.max() + 1e-6)

    def _build_edges(
        self,
        node_feats: torch.Tensor,
        node_time_start: torch.Tensor,
        node_time_end: torch.Tensor,
        node_surprise: torch.Tensor,
        node_merge_count: torch.Tensor,
    ) -> GraphState:
        n = node_feats.size(0)
        device = node_feats.device
        if n <= 1:
            empty = torch.empty(2, 0, device=device, dtype=torch.long)
            return GraphState(
                node_feats=node_feats,
                node_time_start=node_time_start,
                node_time_end=node_time_end,
                node_surprise=node_surprise,
                node_merge_count=node_merge_count,
                edge_index=empty,
                edge_type=torch.empty(0, device=device, dtype=torch.long),
                edge_weight=torch.empty(0, device=device, dtype=node_feats.dtype),
            )

        # Temporal edges by sorted time.
        node_time_center = 0.5 * (node_time_start + node_time_end)
        order = torch.argsort(node_time_center, descending=False)
        t_src = order[:-1]
        t_dst = order[1:]
        temporal_w = torch.exp(-self.temporal_weight_c * node_surprise[t_dst])

        # Similarity edges.
        x = F.normalize(node_feats, dim=-1)
        sim = x @ x.t()
        sim.fill_diagonal_(-1e4)
        k = min(self.sim_topk, max(1, n - 1))
        topv, topi = torch.topk(sim, k=k, dim=-1)
        s_src = torch.arange(n, device=device).unsqueeze(1).expand(n, k).reshape(-1)
        s_dst = topi.reshape(-1)
        sim_w = topv.reshape(-1)

        edge_src = torch.cat([t_src, s_src], dim=0)
        edge_dst = torch.cat([t_dst, s_dst], dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_type = torch.cat(
            [torch.zeros(t_src.size(0), device=device, dtype=torch.long), torch.ones(s_src.size(0), device=device, dtype=torch.long)],
            dim=0,
        )
        edge_weight = torch.cat([temporal_w, sim_w], dim=0)

        return GraphState(
            node_feats=node_feats,
            node_time_start=node_time_start,
            node_time_end=node_time_end,
            node_surprise=node_surprise,
            node_merge_count=node_merge_count,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
        )

    def _collect_hops(
        self,
        edge_index: torch.Tensor,
        seeds: torch.Tensor,
        max_nodes: int,
        node_scores: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Query-conditioned neighborhood expansion. Top-k seeds initialize the
        # route; if the hop frontier exceeds budget, we keep the highest-scoring
        # candidates rather than taking arbitrary graph neighbors.
        if edge_index.numel() == 0 or seeds.numel() == 0:
            if seeds.numel() <= max_nodes:
                return seeds
            if node_scores is None:
                return seeds[:max_nodes]
            keep = torch.topk(node_scores[seeds], k=max_nodes, largest=True).indices
            return seeds[keep]
        src, dst = edge_index[0], edge_index[1]
        selected = seeds.clone()
        if selected.numel() > max_nodes:
            if node_scores is None:
                selected = selected[:max_nodes]
            else:
                keep = torch.topk(node_scores[selected], k=max_nodes, largest=True).indices
                selected = selected[keep]
        frontier = seeds.clone()
        for _ in range(self.num_hops):
            if frontier.numel() == 0:
                break
            mask = torch.isin(src, frontier) | torch.isin(dst, frontier)
            nei = torch.cat([src[mask], dst[mask]], dim=0)
            selected = torch.unique(torch.cat([selected, nei], dim=0))
            if selected.numel() >= max_nodes:
                if node_scores is None:
                    selected = selected[:max_nodes]
                else:
                    # psi_ij(q) = score_j + alpha_r * w_ij (Eq. 7), with w_ij the normalized edge support.
                    route_score = node_scores[selected]
                    if edge_weight is not None and mask.any():
                        edge_bonus = torch.zeros_like(node_scores)
                        edge_src = src[mask]
                        edge_dst = dst[mask]
                        edge_w = edge_weight[mask].to(dtype=edge_bonus.dtype)
                        edge_bonus.index_reduce_(0, edge_src, edge_w, reduce="amax", include_self=True)
                        edge_bonus.index_reduce_(0, edge_dst, edge_w, reduce="amax", include_self=True)
                        route_score = route_score + self.route_edge_weight * edge_bonus[selected]
                    keep = torch.topk(route_score, k=max_nodes, largest=True).indices
                    selected = selected[keep]
                break
            frontier = torch.unique(nei)
        return selected

    def _subgraph(self, graph: GraphState, node_ids: torch.Tensor) -> GraphState:
        if node_ids.numel() == 0:
            return graph
        mapping = -torch.ones(graph.node_feats.size(0), device=graph.node_feats.device, dtype=torch.long)
        mapping[node_ids] = torch.arange(node_ids.size(0), device=graph.node_feats.device)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        valid = (mapping[src] >= 0) & (mapping[dst] >= 0)
        sub_src = mapping[src[valid]]
        sub_dst = mapping[dst[valid]]
        sub_edge_index = torch.stack([sub_src, sub_dst], dim=0) if valid.any() else torch.empty(2, 0, device=graph.node_feats.device, dtype=torch.long)

        return GraphState(
            node_feats=graph.node_feats[node_ids],
            node_time_start=graph.node_time_start[node_ids],
            node_time_end=graph.node_time_end[node_ids],
            node_surprise=graph.node_surprise[node_ids],
            node_merge_count=graph.node_merge_count[node_ids],
            edge_index=sub_edge_index,
            edge_type=graph.edge_type[valid] if valid.any() else torch.empty(0, device=graph.node_feats.device, dtype=torch.long),
            edge_weight=graph.edge_weight[valid] if valid.any() else torch.empty(0, device=graph.node_feats.device, dtype=graph.node_feats.dtype),
        )

    def _evidence_redundancy_loss(self, evidence_tokens: torch.Tensor) -> torch.Tensor:
        if evidence_tokens.size(0) <= 1:
            return evidence_tokens.new_zeros(())
        normed = F.normalize(evidence_tokens, dim=-1)
        sim = normed @ normed.t()
        off_diag = ~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        redundant = (sim.masked_select(off_diag) - self.redundancy_margin).clamp_min(0.0)
        if redundant.numel() == 0:
            return sim.new_zeros(())
        return redundant.mean()

    def forward(
        self,
        node_feats: torch.Tensor,
        node_time_start: torch.Tensor,
        node_time_end: torch.Tensor,
        node_surprise: torch.Tensor,
        node_indices: torch.Tensor,
        query: torch.Tensor,
        node_merge_count: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if node_feats.numel() == 0:
            empty_slots = node_feats.new_zeros(0, self.hidden_size)
            empty_spans = torch.empty(0, 2, dtype=node_feats.dtype, device=node_feats.device)
            return {
                "evidence_tokens": empty_slots,
                "evidence_indices": torch.empty(0, dtype=torch.long, device=node_feats.device),
                "evidence_times": torch.empty(0, dtype=node_feats.dtype, device=node_feats.device),
                "evidence_time_spans": empty_spans,
                "node_scores": torch.empty(0, dtype=node_feats.dtype, device=node_feats.device),
                "node_times": torch.empty(0, dtype=node_feats.dtype, device=node_feats.device),
                "node_time_spans": empty_spans,
                "node_merge_count": torch.empty(0, dtype=node_feats.dtype, device=node_feats.device),
                "node_span_lengths": torch.empty(0, dtype=node_feats.dtype, device=node_feats.device),
                "subgraph_size": torch.tensor(0.0, device=node_feats.device),
                "sparsity_subgraph_loss": torch.tensor(0.0, device=node_feats.device),
                "sparsity_redundancy_loss": torch.tensor(0.0, device=node_feats.device),
                "sparsity_loss": torch.tensor(0.0, device=node_feats.device),
            }

        if node_merge_count is None:
            node_merge_count = node_feats.new_zeros(node_feats.size(0))

        if edge_index is not None and edge_type is not None and edge_weight is not None:
            graph = GraphState(
                node_feats=node_feats,
                node_time_start=node_time_start,
                node_time_end=node_time_end,
                node_surprise=node_surprise,
                node_merge_count=node_merge_count,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_weight=edge_weight,
            )
        else:
            graph = self._build_edges(
                node_feats=node_feats,
                node_time_start=node_time_start,
                node_time_end=node_time_end,
                node_surprise=node_surprise,
                node_merge_count=node_merge_count,
            )

        q = query
        if q.dim() == 2:
            q = q.mean(dim=0)
        q = F.normalize(q, dim=-1)
        nodes_norm = F.normalize(graph.node_feats, dim=-1)
        # Eq. (6): l_i = t_end - t_start + 1; surprise, span and merge count are
        # normalized over the active memory and reused when re-scoring refined nodes.
        span_len = (graph.node_time_end - graph.node_time_start).clamp_min(0) + 1.0
        score_bias = (
            self.eta_surprise_boost * self._normalize_stat(graph.node_surprise)
            - self.span_penalty_weight * self._normalize_stat(span_len)
            - self.merge_penalty_weight * self._normalize_stat(graph.node_merge_count)
        ).to(dtype=graph.node_feats.dtype)
        raw_score = (nodes_norm @ q) + score_bias
        # Routing uses edge supports in [0, 1]: similarity edges map cos to (1 + cos) / 2.
        route_support = torch.where(
            graph.edge_type == 1,
            0.5 * (1.0 + graph.edge_weight),
            graph.edge_weight,
        )
        route_logit = (raw_score - self.route_threshold) / max(self.route_temperature, 1e-6)
        route_prob = torch.sigmoid(route_logit)

        k = min(self.seed_topk, raw_score.numel())
        seeds = torch.topk(raw_score, k=k, largest=True).indices
        selected = self._collect_hops(
            graph.edge_index,
            seeds,
            max_nodes=self.graph_budget,
            node_scores=raw_score,
            edge_weight=route_support,
        )
        sub = self._subgraph(graph, selected)
        sub_node_indices = node_indices[selected]
        refined = self.gar(sub)

        refined_norm = F.normalize(refined, dim=-1)
        sub_span_len = span_len[selected]
        refined_score = (refined_norm @ q) + score_bias[selected].to(dtype=refined.dtype)
        m = min(self.evidence_tokens, refined.size(0))
        ev_pos = torch.topk(refined_score, k=m, largest=True).indices

        # Eq. (10): e_m = LN(W_e h_m). Only the retrieved nodes are exposed (no padding tokens).
        selected_evidence = self.evidence_ln(self.evidence_proj(refined[ev_pos]))
        evidence_tokens = selected_evidence

        evidence_indices = sub_node_indices[ev_pos]
        sparsity_subgraph_loss = route_prob.sum().to(dtype=refined.dtype) / float(max(1, self.graph_budget))
        sparsity_redundancy_loss = self._evidence_redundancy_loss(selected_evidence)
        sparsity_loss = (
            self.spar_subgraph_weight * sparsity_subgraph_loss
            + self.spar_redundancy_weight * sparsity_redundancy_loss
        )
        sub_centers = 0.5 * (sub.node_time_start + sub.node_time_end)
        node_time_spans = torch.stack([sub.node_time_start, sub.node_time_end], dim=-1)
        evidence_time_spans = torch.stack([sub.node_time_start[ev_pos], sub.node_time_end[ev_pos]], dim=-1)

        return {
            "evidence_tokens": evidence_tokens,
            "evidence_indices": evidence_indices,
            "evidence_times": sub_centers[ev_pos],
            "evidence_time_spans": evidence_time_spans,
            "node_indices": sub_node_indices,
            "node_scores": refined_score,
            "node_times": sub_centers,
            "node_time_spans": node_time_spans,
            "node_merge_count": sub.node_merge_count,
            "node_span_lengths": sub_span_len,
            "subgraph_size": torch.tensor(float(selected.size(0)), device=node_feats.device),
            "sparsity_subgraph_loss": sparsity_subgraph_loss,
            "sparsity_redundancy_loss": sparsity_redundancy_loss,
            "sparsity_loss": sparsity_loss.to(device=node_feats.device, dtype=node_feats.dtype),
        }
