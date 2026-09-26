from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from main.cli.common import build_vismem_config, load_yaml
from main.data.collate import load_image
from main.data.jsonl_dataset import JsonlVLDataset, Sample
from main.model.model import VisMemModel
from main.trainer.method_sft import method_sft_loss
from main.utils.logging import get_logger
from main.utils.misc import ensure_dir, set_seed, to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl
from main.utils.video import load_video_stream_units

logger = get_logger("main.train_method_sft")


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _resolve_sample_fps(sample: Sample, default_sample_fps: float) -> float:
    meta = sample.meta or {}
    sample_fps = _maybe_float(sample.sample_fps)
    if sample_fps is None:
        sample_fps = _maybe_float(meta.get("sample_fps", None))
    if sample_fps is None:
        chunk_duration = _maybe_float(meta.get("chunk_duration", meta.get("sample_stride_sec", None)))
        if chunk_duration is not None and chunk_duration > 0:
            sample_fps = 1.0 / chunk_duration
    if sample_fps is None:
        sample_fps = float(default_sample_fps)
    if sample_fps <= 0:
        raise ValueError(f"Resolved `sample_fps` must be > 0, got {sample_fps}.")
    return sample_fps


def _resolve_chunk_duration(sample: Sample, sample_fps: float) -> float:
    meta = sample.meta or {}
    chunk_duration = _maybe_float(meta.get("chunk_duration", meta.get("clip_duration", None)))
    if chunk_duration is None:
        chunk_duration = 1.0 / max(sample_fps, 1e-6)
    if chunk_duration <= 0:
        raise ValueError(f"Resolved `chunk_duration` must be > 0, got {chunk_duration}.")
    return chunk_duration


def _resolve_clip_frames(sample: Sample) -> int:
    meta = sample.meta or {}
    clip_frames = _maybe_int(meta.get("clip_frames", meta.get("frames_per_clip", 4)))
    if clip_frames is None:
        clip_frames = 4
    if clip_frames <= 0:
        raise ValueError(f"Resolved `clip_frames` must be > 0, got {clip_frames}.")
    return clip_frames


def _resolve_stream_update_unit(sample: Sample, default_stream_update_unit: str) -> str:
    meta = sample.meta or {}
    unit = str(
        meta.get(
            "stream_update_unit",
            meta.get("stream_unit", default_stream_update_unit),
        )
    ).strip().lower()
    if unit not in {"clip", "frame"}:
        raise ValueError(f"Resolved `stream_update_unit` must be 'clip' or 'frame', got {unit!r}.")
    return unit


def _prepare_streaming_sample(
    sample: Sample,
    model: VisMemModel,
    state_dtype: torch.dtype,
    default_sample_fps: float,
    default_history_prompt: str,
    default_stream_max_frames: Optional[int],
    default_stream_update_unit: str,
):
    if not sample.video:
        raise ValueError("Streaming sample must contain `video`.")

    meta = dict(sample.meta or {})
    sample_fps = _resolve_sample_fps(sample, default_sample_fps)
    stream_update_unit = _resolve_stream_update_unit(sample, default_stream_update_unit)
    chunk_duration = _resolve_chunk_duration(sample, sample_fps) if stream_update_unit == "clip" else None
    clip_frames = _resolve_clip_frames(sample) if stream_update_unit == "clip" else 1
    clip_layout = str(meta.get("clip_layout", "grid")) if stream_update_unit == "clip" else "single"
    question_time = _maybe_float(sample.question_time)
    if question_time is None:
        question_time = _maybe_float(meta.get("question_time", meta.get("query_time", None)))

    start_time = _maybe_float(sample.start_time)
    if start_time is None:
        start_time = _maybe_float(meta.get("start_time", None))
    if start_time is None:
        start_time = 0.0

    end_time = question_time
    if end_time is None:
        end_time = _maybe_float(sample.end_time)
    if end_time is None:
        end_time = _maybe_float(meta.get("end_time", None))

    max_frames = _maybe_int(sample.max_frames)
    if max_frames is None:
        max_frames = _maybe_int(meta.get("max_frames", default_stream_max_frames))

    history_prompt = str(meta.get("history_prompt", default_history_prompt))
    sampled = load_video_stream_units(
        video_path=sample.video,
        sample_fps=sample_fps,
        unit=stream_update_unit,
        clip_duration=chunk_duration,
        clip_frames=clip_frames,
        max_units=max_frames,
        start_time=start_time,
        end_time=end_time,
        compose_mode=clip_layout,
    )
    if not sampled:
        raise ValueError(
            f"No stream units sampled for streaming sample id={sample.id!r}, video={sample.video!r}, "
            f"start_time={start_time}, end_time={end_time}, sample_fps={sample_fps}."
        )

    memory_state = model.init_memory_state(batch_size=1, mem_type="short", dtype=state_dtype)
    if memory_state is None:
        raise ValueError("Current backend does not support stateful LVM memory.")

    first_span = sampled[0][1]
    memory_state[0].next_step = float(first_span[1])
    prev_time = None
    history_units = sampled[:-1]
    for unit_img, (unit_start, unit_end) in history_units:
        step_stride = max(float(unit_end - prev_time), 1e-3) if prev_time is not None else max(float(unit_end - unit_start), 1e-3)
        # Keep historical writes inside the training graph so answer loss can
        # teach the model how to write better memory, not only how to read it.
        memory_state = model.update_memory_with_grad(
            images=[unit_img],
            prompts=[history_prompt],
            memory_state=memory_state,
            return_memory_state=True,
            step_stride=step_stride,
            chunk_time_spans=torch.tensor([[unit_start, unit_end]], device=model.device, dtype=torch.float32),
        )
        prev_time = float(unit_end)

    query_image, (query_start, query_end) = sampled[-1]
    query_time = float(query_end)
    query_step_stride = max(float(query_end - prev_time), 1e-3) if prev_time is not None else max(float(query_end - query_start), 1e-3)

    meta.setdefault("query_time", float(query_time))
    meta.setdefault("question_time", float(query_time if question_time is None else question_time))
    meta.setdefault("sample_fps", float(sample_fps))
    meta.setdefault("stream_update_unit", stream_update_unit)
    meta.setdefault("history_units", len(history_units))
    if chunk_duration is not None:
        meta.setdefault("chunk_duration", float(chunk_duration))
    if stream_update_unit == "clip":
        meta.setdefault("clip_frames", int(clip_frames))
        meta.setdefault("clip_layout", clip_layout)
    meta.setdefault("history_frames", len(history_units))
    if max_frames is not None:
        meta.setdefault("max_frames", int(max_frames))

    return {
        "image": query_image,
        "prompt": sample.prompt,
        "answer": sample.answer,
        "meta": meta,
        "memory_state": memory_state,
        "step_stride": query_step_stride,
        "chunk_time_span": torch.tensor([[query_start, query_end]], device=model.device, dtype=torch.float32),
        "history_frames": len(history_units),
        "query_time": float(query_time),
        "mode": "video",
    }


def _prepare_sample(
    sample: Sample,
    model: VisMemModel,
    state_dtype: torch.dtype,
    default_sample_fps: float,
    default_history_prompt: str,
    default_stream_max_frames: Optional[int],
    default_stream_update_unit: str,
):
    if sample.is_streaming:
        return _prepare_streaming_sample(
            sample=sample,
            model=model,
            state_dtype=state_dtype,
            default_sample_fps=default_sample_fps,
            default_history_prompt=default_history_prompt,
            default_stream_max_frames=default_stream_max_frames,
            default_stream_update_unit=default_stream_update_unit,
        )

    image = load_image(sample.image)
    if image is None:
        raise ValueError(f"Image sample id={sample.id!r} does not contain a valid `image` path.")
    return {
        "image": image,
        "prompt": sample.prompt,
        "answer": sample.answer,
        "meta": dict(sample.meta or {}),
        "memory_state": None,
        "step_stride": 1.0,
        "chunk_time_span": None,
        "history_frames": 0,
        "query_time": None,
        "mode": "image",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selectstream_qwen25vl7b_lvm_sft.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--init_from", default=None, help="Checkpoint folder containing main.pt to continue training.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--beta_ret", type=float, default=None)
    ap.add_argument("--gamma_spar", type=float, default=None)
    ap.add_argument("--ret_id_weight", type=float, default=None)
    ap.add_argument("--ret_time_weight", type=float, default=None)
    ap.add_argument("--time_tolerance", type=float, default=None)
    ap.add_argument("--evidence_topk", type=int, default=None)
    ap.add_argument("--default_sample_fps", type=float, default=1.0, help="Fallback fps for streaming video samples.")
    ap.add_argument("--stream_max_frames", type=int, default=None, help="Optional cap on sampled frames per streaming sample.")
    ap.add_argument("--stream_update_unit", default="frame", choices=["clip", "frame"], help="Streaming update granularity. 'frame' updates SAW/LVM on every sampled frame.")
    ap.add_argument("--history_prompt", default="", help="Prompt used when feeding history chunks into memory.")
    ap.add_argument("--freeze_lvm_writer", action="store_true", help="Freeze LVM writer modules in this run.")
    ap.add_argument("--freeze_graph_reasoner", action="store_true", help="Freeze DMG/GAR modules in this run.")
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_vismem_config(cfg_dict)

    train_cfg = cfg_dict.get("training", {})
    set_seed(int(train_cfg.get("seed", 42)))

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map = cfg_dict["model"].get("device_map", "auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust,
    )
    model = VisMemModel(base_model, tokenizer, processor, viscfg)

    if args.init_from is not None:
        ckpt_path = os.path.join(args.init_from, "main.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Cannot find checkpoint file: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        payload = state.get("vismem_state", state)
        incompatible = model.load_state_dict(payload, strict=False)
        logger.info(
            "Loaded init checkpoint from %s (missing=%d, unexpected=%d).",
            ckpt_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    # The backbone VLM stays frozen; only the SelectStream modules are trained (Sec. 3.6).
    for p in model.base_model.parameters():
        p.requires_grad = False

    freeze_lvm_writer = bool(args.freeze_lvm_writer or train_cfg.get("freeze_lvm_writer", False))
    freeze_graph_reasoner = bool(args.freeze_graph_reasoner or train_cfg.get("freeze_graph_reasoner", False))

    if freeze_lvm_writer:
        for former in (model.lvm_short_former, model.lvm_long_former):
            if former is None:
                continue
            for p in former.parameters():
                p.requires_grad = False

    if freeze_graph_reasoner and model.graph_reasoner is not None:
        for p in model.graph_reasoner.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters left. Check freeze flags/config.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    logger.info(
        "Trainable params: %d / %d (%.2f%%), freeze_lvm_writer=%s, freeze_graph_reasoner=%s",
        trainable_params,
        total_params,
        100.0 * float(trainable_params) / float(max(1, total_params)),
        str(freeze_lvm_writer),
        str(freeze_graph_reasoner),
    )

    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 2e-4))
    beta_ret = args.beta_ret if args.beta_ret is not None else float(train_cfg.get("beta_ret", 0.1))
    gamma_spar = args.gamma_spar if args.gamma_spar is not None else float(train_cfg.get("gamma_spar", 0.05))
    ret_id_weight = args.ret_id_weight if args.ret_id_weight is not None else float(train_cfg.get("ret_id_weight", 1.0))
    ret_time_weight = args.ret_time_weight if args.ret_time_weight is not None else float(train_cfg.get("ret_time_weight", 1.0))
    time_tolerance = args.time_tolerance if args.time_tolerance is not None else float(train_cfg.get("time_tolerance", 0.0))
    evidence_topk = args.evidence_topk if args.evidence_topk is not None else int(train_cfg.get("evidence_topk", 8))
    grad_accum = max(1, args.grad_accum if args.grad_accum is not None else int(train_cfg.get("grad_accum", 4)))
    opt = optim.AdamW(trainable, lr=lr)

    ds = JsonlVLDataset(args.train_jsonl)
    ensure_dir(args.output_dir)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(range(len(ds)), desc=f"Method-SFT epoch {epoch}")
        opt.zero_grad()
        accum_steps = 0
        for i in pbar:
            sample = ds[i]
            if sample.answer is None:
                continue
            prepared = _prepare_sample(
                sample=sample,
                model=model,
                state_dtype=dtype,
                default_sample_fps=args.default_sample_fps,
                default_history_prompt=args.history_prompt,
                default_stream_max_frames=args.stream_max_frames,
                default_stream_update_unit=args.stream_update_unit,
            )
            img = prepared["image"]
            prompt = prepared["prompt"]
            answer = prepared["answer"]
            meta = prepared["meta"]
            memory_state = prepared["memory_state"]
            step_stride = prepared["step_stride"]
            chunk_time_span = prepared["chunk_time_span"]

            inputs = model.prepare_inputs([img], [prompt])

            loss, stats = method_sft_loss(
                base_model=model.base_model,
                selectstream_model=model,
                inputs=inputs,
                target_text=answer,
                meta=meta,
                memory_state=memory_state,
                step_stride=step_stride,
                chunk_time_spans=chunk_time_span,
                beta_ret=beta_ret,
                gamma_spar=gamma_spar,
                ret_id_weight=ret_id_weight,
                ret_time_weight=ret_time_weight,
                time_tolerance=time_tolerance,
                evidence_topk=evidence_topk,
            )

            (loss / grad_accum).backward()
            accum_steps += 1
            if accum_steps % grad_accum == 0:
                opt.step()
                opt.zero_grad()

            postfix = {
                "L": float(stats["loss_total"].cpu()),
                "ans": float(stats["loss_ans"].cpu()),
                "ret": float(stats["loss_ret"].cpu()),
                "rid": float(stats["loss_ret_id"].cpu()),
                "rtime": float(stats["loss_ret_time"].cpu()),
                "spar": float(stats["loss_spar"].cpu()),
                "hid": float(stats["hit_id_at_k"].cpu()),
                "ht": float(stats["hit_time_at_k"].cpu()),
                "mode": prepared["mode"],
            }
            if "subgraph_size" in stats:
                postfix["sg"] = float(stats["subgraph_size"].cpu())
            if "loss_spar_subgraph" in stats:
                postfix["ssg"] = round(float(stats["loss_spar_subgraph"].cpu()), 4)
            if "loss_spar_redundancy" in stats:
                postfix["sred"] = round(float(stats["loss_spar_redundancy"].cpu()), 4)
            lvm_stats = model.get_lvm_debug_stats(mem_type="short")
            if lvm_stats is not None:
                postfix["seg"] = int(lvm_stats["num_segments"])
                postfix["new"] = int(lvm_stats["num_new"])
                postfix["upd"] = int(lvm_stats["num_update"])
                postfix["mrg"] = int(lvm_stats["num_merge"])
                postfix["mpen"] = round(float(lvm_stats["merge_penalty_total"]), 4)
            if prepared["mode"] == "video":
                postfix["hist"] = int(prepared["history_frames"])
                postfix["qt"] = round(float(prepared["query_time"]), 1)
            pbar.set_postfix(postfix)

        if accum_steps % grad_accum != 0:
            opt.step()
            opt.zero_grad()

        ckpt = os.path.join(args.output_dir, f"epoch{epoch}")
        ensure_dir(ckpt)
        torch.save({"vismem_state": model.state_dict(), "config": cfg_dict}, os.path.join(ckpt, "main.pt"))
        tokenizer.save_pretrained(ckpt)

    logger.info("Method-SFT done.")


if __name__ == "__main__":
    main()
