from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset


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


@dataclass
class Sample:
    id: str
    image: Optional[str]
    prompt: str
    video: Optional[str] = None
    answer: Optional[str] = None
    question_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    sample_fps: Optional[float] = None
    max_frames: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def is_streaming(self) -> bool:
        return self.video is not None

class JsonlVLDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Sample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", obj.get("question", None))
                if prompt is None:
                    raise KeyError("Each JSONL row must contain `prompt` or `question`.")
                self.items.append(
                    Sample(
                        id=str(obj.get("id", len(self.items))),
                        image=obj.get("image", None),
                        prompt=prompt,
                        video=obj.get("video", None),
                        answer=obj.get("answer", None),
                        question_time=_maybe_float(obj.get("question_time", obj.get("query_time", None))),
                        start_time=_maybe_float(obj.get("start_time", None)),
                        end_time=_maybe_float(obj.get("end_time", None)),
                        sample_fps=_maybe_float(obj.get("sample_fps", None)),
                        max_frames=_maybe_int(obj.get("max_frames", None)),
                        meta={k:v for k,v in obj.items() if k not in ("id","image","video","prompt","question","answer")}
                    )
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]
