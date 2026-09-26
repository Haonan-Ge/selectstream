from __future__ import annotations
import argparse
import os
from tqdm import tqdm
from PIL import Image
import torch

from main.utils.logging import get_logger
from main.utils.misc import to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl
from main.model.model import VisMemModel
from main.data.jsonl_dataset import JsonlVLDataset
from main.data.collate import collate_samples
from main.trainer.rewards import exact_match_reward, substring_reward
from main.trainer.method_sft import (
    _extract_evidence_ids,
    _extract_evidence_time_intervals,
    _hit_at_k,
    _temporal_overlap,
)
from main.cli.common import load_yaml, build_vismem_config
from main.utils.streaming import run_streaming_query
from main.utils.video import load_video_stream_units

logger = get_logger("main.eval")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selectstream_qwen25vl7b_baseline.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--ckpt", default=None, help="folder with main.pt")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--enable_vismem", action="store_true")
    ap.add_argument("--method_infer", action="store_true", help="Use method-style query->graph retrieval inference.")
    ap.add_argument("--report_evidence", action="store_true", help="Report evidence hit metrics when metadata is available.")
    ap.add_argument("--evidence_topk", type=int, default=8)
    ap.add_argument("--metric", choices=["exact","substr"], default="substr")
    ap.add_argument("--default_sample_fps", type=float, default=1.0, help="Stream sampling rate for video samples.")
    ap.add_argument("--stream_update_unit", default="frame", choices=["clip", "frame"])
    ap.add_argument("--stream_max_frames", type=int, default=None)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_vismem_config(cfg_dict)

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype","bfloat16"))
    device_map = cfg_dict["model"].get("device_map","auto")
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust)
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)
    if args.ckpt is not None:
        state = torch.load(os.path.join(args.ckpt, "main.pt"), map_location="cpu")
        vismem.load_state_dict(state["vismem_state"], strict=False)
    vismem.eval()

    ds = JsonlVLDataset(args.jsonl)
    preds, refs = [], []
    hit_id_scores = []
    hit_time_scores = []
    t_overlap_scores = []
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        batch = collate_samples([sample])
        img = batch["images"][0]
        prompt = batch["prompts"][0]
        answer = batch["answers"][0]
        meta = batch["metas"][0]
        if answer is None:
            continue
        sample_fps = sample.sample_fps or float((meta or {}).get("sample_fps", args.default_sample_fps))
        if args.method_infer and sample.is_streaming:
            # Causal protocol: only the prefix observed up to the question time is used.
            units = load_video_stream_units(
                video_path=sample.video,
                sample_fps=sample_fps,
                unit=str((meta or {}).get("stream_update_unit", args.stream_update_unit)),
                max_units=sample.max_frames or args.stream_max_frames,
                start_time=sample.start_time or 0.0,
                end_time=sample.question_time if sample.question_time is not None else sample.end_time,
            )
            if not units:
                logger.warning("No stream units sampled for id=%s; skipping.", sample.id)
                continue
            pred_list, aux, _ = run_streaming_query(
                vismem,
                units,
                prompt=prompt,
                dtype=dtype,
                max_new_tokens=args.max_new_tokens,
            )
            pred = pred_list[0]
        elif args.method_infer:
            pred_list, aux = vismem.generate_method(
                images=[img],
                prompts=[prompt],
                max_new_tokens=args.max_new_tokens,
                return_aux=True,
            )
            pred = pred_list[0]
        if args.method_infer:
            if args.report_evidence and aux:
                aux0 = aux[0]
                node_indices = aux0.get("node_indices", [])
                node_scores = aux0.get("node_scores", [])
                node_times = aux0.get("node_time_spans", aux0.get("node_times", []))
                gt_ids = _extract_evidence_ids(meta)
                gt_times = _extract_evidence_time_intervals(meta)

                if node_indices and node_scores and gt_ids:
                    hit_id_scores.append(
                        _hit_at_k(
                            torch.tensor(node_indices, dtype=torch.long),
                            torch.tensor(node_scores, dtype=torch.float32),
                            gt_ids,
                            args.evidence_topk,
                        )
                    )
                if gt_times:
                    # Recall@M and T-Overlap over the M injected evidence nodes (Appendix E).
                    overlap = _temporal_overlap(
                        torch.tensor(aux0.get("evidence_time_spans", []), dtype=torch.float32),
                        gt_times,
                        frame_duration=1.0 / sample_fps,
                    )
                    hit_time_scores.append(float(overlap > 0))
                    t_overlap_scores.append(overlap)
        else:
            pred = vismem.generate(
                images=[img],
                prompts=[prompt],
                max_new_tokens=args.max_new_tokens,
                enable_vismem=args.enable_vismem,
            )[0]
        preds.append(pred)
        refs.append(answer)

    if args.metric == "exact":
        rewards = exact_match_reward(preds, refs)
    else:
        rewards = substring_reward(preds, refs)
    score = sum(rewards) / max(1, len(rewards))
    print(f"{args.metric} score: {score:.4f} ({len(rewards)} examples)")
    if hit_id_scores:
        print(f"evidence hit@{args.evidence_topk} (id): {sum(hit_id_scores) / len(hit_id_scores):.4f} ({len(hit_id_scores)} examples)")
    if hit_time_scores:
        print(f"Recall@M: {sum(hit_time_scores) / len(hit_time_scores):.4f} ({len(hit_time_scores)} examples)")
    if t_overlap_scores:
        print(f"T-Overlap: {sum(t_overlap_scores) / len(t_overlap_scores):.4f} ({len(t_overlap_scores)} examples)")

if __name__ == "__main__":
    main()
