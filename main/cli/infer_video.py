from __future__ import annotations

import argparse
import os

import torch

from main.cli.common import build_vismem_config, load_yaml
from main.model.model import VisMemModel
from main.utils.logging import get_logger
from main.utils.misc import to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl
from main.utils.streaming import run_streaming_query
from main.utils.video import load_video_stream_units

logger = get_logger("main.infer_video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selectstream_qwen25vl7b_lvm_sft.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--ckpt", default=None, help="folder with main.pt")
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--sample_fps", type=float, default=1.0, help="Sampling rate for streaming frames.")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--stream_update_unit", default="frame", choices=["clip", "frame"], help="Streaming update granularity. 'frame' updates SAW/LVM on every sampled frame.")
    ap.add_argument("--chunk_duration", type=float, default=None, help="Temporal duration of each streaming clip.")
    ap.add_argument("--clip_frames", type=int, default=4, help="How many raw frames to compose into one streaming clip.")
    ap.add_argument("--clip_layout", default="grid", choices=["grid", "concat_h", "concat_v"])
    ap.add_argument("--start_time", type=float, default=0.0)
    ap.add_argument("--end_time", type=float, default=None)
    ap.add_argument("--query_frame_index", type=int, default=-1, help="Which sampled stream unit to answer on. Default: last sampled unit.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--show_evidence", action="store_true")
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path

    viscfg = build_vismem_config(cfg_dict)
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
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)
    if args.ckpt is not None:
        state = torch.load(os.path.join(args.ckpt, "main.pt"), map_location="cpu")
        vismem.load_state_dict(state["vismem_state"], strict=False)
    vismem.eval()

    sampled = load_video_stream_units(
        video_path=args.video,
        sample_fps=args.sample_fps,
        unit=args.stream_update_unit,
        clip_duration=args.chunk_duration,
        clip_frames=args.clip_frames,
        max_units=args.max_frames,
        start_time=args.start_time,
        end_time=args.end_time,
        compose_mode=args.clip_layout,
    )
    if not sampled:
        raise ValueError("No stream units were sampled from the video. Check `sample_fps`, `stream_update_unit`, `chunk_duration`, `start_time`, and `end_time`.")

    query_idx = args.query_frame_index if args.query_frame_index >= 0 else len(sampled) - 1
    if query_idx < 0 or query_idx >= len(sampled):
        raise ValueError(f"`query_frame_index` out of range: {query_idx}, sampled units={len(sampled)}")

    logger.info("Sampled %d %s units from %s. Query unit index=%d", len(sampled), args.stream_update_unit, args.video, query_idx)
    texts, aux, _ = run_streaming_query(
        vismem,
        sampled,
        prompt=args.prompt,
        query_idx=query_idx,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    query_time = float(sampled[query_idx][1][1])

    print(texts[0])
    if args.show_evidence and aux:
        print("query_time:", query_time)
        print("evidence:", aux[0])


if __name__ == "__main__":
    main()
