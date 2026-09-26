"""
Convert Streamo-Instruct-465K annotations into SelectStream streaming JSONL.

Streamo raw records look like:
    {"video_path": "...", "task_type": "QA", "source": "...",
     "question": [{"content": "...", "time": "5"}],
     "response": [{"content": "...", "st_time": 5.0, "end_time": 6.0, "time": ""}]}

Streamo time `t` denotes the second <t-1 s, t s>, i.e. the 1-fps frame sampled at t - 1.
Each response becomes one causal training row:
  - span response (st_time/end_time): evidence = the event frames, and the question is
    answered at the frame right after the event ends, so the evidence lies in the observed
    history (retrieval supervision via L_ret);
  - instant response (time): answered at that frame, answer loss only.
The prompt is the latest question asked no later than the answer frame.

Usage:
    python -m main.data.prepare_streamo \
        --anno_dir data/Streamo-Instruct-465K \
        --video_root data/videos \
        --output data/streamo_selectstream.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple


def parse_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + float(part)
        return seconds
    except ValueError:
        return None


def streamo_frame(t: float) -> float:
    # Streamo second <t-1, t> -> timestamp of the 1-fps frame sampled at t - 1.
    return max(0.0, t - 1.0)


def convert_record(
    record: Dict[str, Any],
    record_id: str,
    video_root: Optional[str] = None,
    max_history_sec: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    questions: List[Tuple[float, str]] = []
    for q in record.get("question") or []:
        content = str(q.get("content", "")).strip()
        if not content:
            continue
        t = parse_seconds(q.get("time"))
        questions.append((streamo_frame(t) if t is not None else 0.0, content))
    if not questions:
        return
    questions.sort(key=lambda x: x[0])

    video = record.get("video_path") or record.get("video") or ""
    if video_root and not os.path.isabs(video):
        video = os.path.join(video_root, video)

    for r_idx, resp in enumerate(record.get("response") or []):
        answer = str(resp.get("content", "")).strip()
        if not answer:
            continue
        st = parse_seconds(resp.get("st_time"))
        end = parse_seconds(resp.get("end_time"))
        t = parse_seconds(resp.get("time"))

        evidence = None
        if st is not None and end is not None:
            lo, hi = sorted((streamo_frame(st), streamo_frame(end)))
            evidence = [[lo, hi]]
            answer_time = hi + 1.0
        elif t is not None:
            answer_time = streamo_frame(t)
        else:
            continue

        asked = [q for q in questions if q[0] <= answer_time] or questions[:1]
        question_frame, prompt = asked[-1]
        question_time = max(answer_time, question_frame)

        row: Dict[str, Any] = {
            "id": f"{record_id}_r{r_idx}",
            "video": video,
            "question": prompt,
            "question_time": question_time,
            "answer": answer,
            "sample_fps": 1.0,
            "stream_update_unit": "frame",
            "start_time": 0.0 if max_history_sec is None else max(0.0, question_time - max_history_sec),
            "task_type": record.get("task_type"),
            "source": record.get("source"),
        }
        if evidence is not None:
            row["evidence_timestamps"] = evidence
        yield row


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno_dir", required=True, help="Downloaded Streamo-Instruct-465K folder (searched recursively for *.json).")
    ap.add_argument("--video_root", default=None, help="Root that the relative `video_path` fields resolve against.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="Optional subfolders to keep, e.g. qa event_grounding.")
    ap.add_argument("--skip_missing_videos", action="store_true", help="Drop rows whose video file does not exist.")
    ap.add_argument("--max_history_sec", type=float, default=None, help="Optionally start streaming at question_time - max_history_sec.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.anno_dir, "**", "*.json"), recursive=True))
    if args.tasks:
        keep = set(args.tasks)
        files = [f for f in files if os.path.relpath(f, args.anno_dir).split(os.sep)[0] in keep]
    if not files:
        raise FileNotFoundError(f"No annotation *.json files found under {args.anno_dir}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    n_records = n_rows = n_evidence = n_missing = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            prefix = os.path.splitext(os.path.relpath(path, args.anno_dir))[0].replace(os.sep, "/")
            for i, record in enumerate(load_records(path)):
                n_records += 1
                for row in convert_record(record, f"{prefix}/{i}", args.video_root, args.max_history_sec):
                    if args.skip_missing_videos and not os.path.exists(row["video"]):
                        n_missing += 1
                        continue
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
                    n_evidence += int("evidence_timestamps" in row)

    print(
        f"{len(files)} files, {n_records} records -> {n_rows} rows "
        f"({n_evidence} with evidence timestamps, {n_rows - n_evidence} answer-only); "
        f"skipped {n_missing} rows with missing videos. Wrote {args.output}"
    )


if __name__ == "__main__":
    main()
