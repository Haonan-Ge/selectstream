from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from PIL import Image


StreamUnit = Tuple[Image.Image, Tuple[float, float]]


def run_streaming_query(
    model,
    units: Sequence[StreamUnit],
    prompt: str,
    query_idx: int = -1,
    history_prompt: str = "",
    dtype: Optional[torch.dtype] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[List[str], Optional[List[Any]], Any]:
    """
    Causal streaming protocol: every unit before `query_idx` is written into memory online,
    then the query is answered on the current unit plus retrieved latent evidence.
    Returns (texts, aux, memory_state).
    """
    if not units:
        raise ValueError("`units` must contain at least one stream unit.")
    query_idx = query_idx if query_idx >= 0 else len(units) - 1
    if query_idx >= len(units):
        raise ValueError(f"`query_idx` out of range: {query_idx}, units={len(units)}")

    state = model.init_memory_state(batch_size=1, mem_type="short", dtype=dtype)
    if state is None:
        raise ValueError("Current backend does not support stateful LVM memory.")
    state[0].next_step = float(units[0][1][1])

    prev_time = None
    for unit_img, (unit_start, unit_end) in units[:query_idx]:
        step_stride = max(float(unit_end - prev_time), 1e-3) if prev_time is not None else max(float(unit_end - unit_start), 1e-3)
        state = model.update_memory(
            images=[unit_img],
            prompts=[history_prompt],
            memory_state=state,
            return_memory_state=True,
            step_stride=step_stride,
            chunk_time_spans=torch.tensor([[unit_start, unit_end]], device=model.device, dtype=torch.float32),
        )
        prev_time = float(unit_end)

    query_image, (query_start, query_end) = units[query_idx]
    query_step_stride = max(float(query_end - prev_time), 1e-3) if prev_time is not None else max(float(query_end - query_start), 1e-3)
    texts, aux, state = model.generate_method(
        images=[query_image],
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        return_aux=True,
        memory_state=state,
        return_memory_state=True,
        step_stride=query_step_stride,
        chunk_time_spans=torch.tensor([[query_start, query_end]], device=model.device, dtype=torch.float32),
    )
    return texts, aux, state
