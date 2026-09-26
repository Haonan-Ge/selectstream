from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageOps


def _load_video_with_imageio(
    video_path: str,
    sample_fps: float,
    max_frames: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> List[Tuple[Image.Image, float]]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 0.0) or 0.0)
        if fps <= 0:
            raise ValueError(f"Could not infer video fps from {video_path}.")

        stride = max(int(round(fps / sample_fps)), 1)
        start_idx = max(0, int(round(start_time * fps)))
        end_idx = None if end_time is None else max(start_idx, int(round(end_time * fps)))

        out: List[Tuple[Image.Image, float]] = []
        for idx, frame in enumerate(reader):
            if idx < start_idx:
                continue
            if end_idx is not None and idx > end_idx:
                break
            if (idx - start_idx) % stride != 0:
                continue
            img = Image.fromarray(frame).convert("RGB")
            out.append((img, float(idx) / fps))
            if max_frames is not None and len(out) >= max_frames:
                break
        return out
    finally:
        reader.close()


def _load_video_with_cv2(
    video_path: str,
    sample_fps: float,
    max_frames: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> List[Tuple[Image.Image, float]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise ValueError(f"Could not infer video fps from {video_path}.")

        stride = max(int(round(fps / sample_fps)), 1)
        start_idx = max(0, int(round(start_time * fps)))
        end_idx = None if end_time is None else max(start_idx, int(round(end_time * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        out: List[Tuple[Image.Image, float]] = []
        idx = start_idx
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if end_idx is not None and idx > end_idx:
                break
            if (idx - start_idx) % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append((Image.fromarray(frame_rgb).convert("RGB"), float(idx) / fps))
                if max_frames is not None and len(out) >= max_frames:
                    break
            idx += 1
        return out
    finally:
        cap.release()


def load_video_sampled_frames(
    video_path: str,
    sample_fps: float = 0.5,
    max_frames: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> List[Tuple[Image.Image, float]]:
    if sample_fps <= 0:
        raise ValueError(f"`sample_fps` must be > 0, got {sample_fps}.")
    if start_time < 0:
        raise ValueError(f"`start_time` must be >= 0, got {start_time}.")
    if end_time is not None and end_time < start_time:
        raise ValueError(f"`end_time` must be >= `start_time`, got {end_time} < {start_time}.")

    imageio_error = None
    try:
        return _load_video_with_imageio(
            video_path=video_path,
            sample_fps=sample_fps,
            max_frames=max_frames,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:
        imageio_error = exc

    try:
        return _load_video_with_cv2(
            video_path=video_path,
            sample_fps=sample_fps,
            max_frames=max_frames,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as cv2_error:
        raise RuntimeError(
            "Failed to load video. Install `imageio` or `opencv-python`, and verify the video path is valid. "
            f"imageio error: {imageio_error}; cv2 error: {cv2_error}"
        ) from cv2_error


def compose_clip_frames(
    frames: Sequence[Image.Image],
    mode: str = "grid",
    padding: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if len(frames) == 0:
        raise ValueError("`frames` must contain at least one image.")
    if len(frames) == 1:
        return frames[0]

    base_w, base_h = frames[0].size
    resized = [ImageOps.fit(img.convert("RGB"), (base_w, base_h), method=Image.Resampling.BILINEAR) for img in frames]

    if mode == "concat_h":
        canvas = Image.new("RGB", (base_w * len(resized) + padding * (len(resized) - 1), base_h), color=bg_color)
        x = 0
        for img in resized:
            canvas.paste(img, (x, 0))
            x += base_w + padding
        return canvas

    if mode == "concat_v":
        canvas = Image.new("RGB", (base_w, base_h * len(resized) + padding * (len(resized) - 1)), color=bg_color)
        y = 0
        for img in resized:
            canvas.paste(img, (0, y))
            y += base_h + padding
        return canvas

    cols = max(1, int(math.ceil(math.sqrt(len(resized)))))
    rows = int(math.ceil(len(resized) / float(cols)))
    canvas_w = cols * base_w + padding * (cols - 1)
    canvas_h = rows * base_h + padding * (rows - 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    for idx, img in enumerate(resized):
        row = idx // cols
        col = idx % cols
        x = col * (base_w + padding)
        y = row * (base_h + padding)
        canvas.paste(img, (x, y))
    return canvas


def load_video_sampled_clips(
    video_path: str,
    sample_fps: float = 0.5,
    clip_duration: Optional[float] = None,
    clip_frames: int = 4,
    max_clips: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    compose_mode: str = "grid",
) -> List[Tuple[Image.Image, Tuple[float, float]]]:
    if sample_fps <= 0:
        raise ValueError(f"`sample_fps` must be > 0, got {sample_fps}.")
    if clip_frames <= 0:
        raise ValueError(f"`clip_frames` must be > 0, got {clip_frames}.")
    if clip_duration is None:
        clip_duration = 1.0 / sample_fps
    if clip_duration <= 0:
        raise ValueError(f"`clip_duration` must be > 0, got {clip_duration}.")

    step = 1.0 / sample_fps
    intra_fps = clip_frames / max(clip_duration, 1e-6)

    if end_time is None:
        # Reuse the frame loader to infer the final timestamp cheaply enough for this prototype.
        preview = load_video_sampled_frames(
            video_path=video_path,
            sample_fps=max(sample_fps, intra_fps),
            start_time=start_time,
            end_time=None,
        )
        if not preview:
            return []
        end_time = float(preview[-1][1])

    clips: List[Tuple[Image.Image, Tuple[float, float]]] = []
    clip_start = float(start_time)
    while clip_start <= float(end_time) + 1e-6:
        clip_end = min(clip_start + float(clip_duration), float(end_time))
        frames_with_ts = load_video_sampled_frames(
            video_path=video_path,
            sample_fps=max(sample_fps, intra_fps),
            max_frames=clip_frames,
            start_time=clip_start,
            end_time=clip_end,
        )
        if frames_with_ts:
            clip_img = compose_clip_frames([img for img, _ in frames_with_ts], mode=compose_mode)
            clips.append((clip_img, (clip_start, clip_end)))
            if max_clips is not None and len(clips) >= max_clips:
                break
        clip_start += step
    return clips


def load_video_stream_units(
    video_path: str,
    sample_fps: float = 0.5,
    unit: str = "clip",
    clip_duration: Optional[float] = None,
    clip_frames: int = 4,
    max_units: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    compose_mode: str = "grid",
) -> List[Tuple[Image.Image, Tuple[float, float]]]:
    unit = str(unit).strip().lower()
    if unit == "frame":
        frames_with_ts = load_video_sampled_frames(
            video_path=video_path,
            sample_fps=sample_fps,
            max_frames=max_units,
            start_time=start_time,
            end_time=end_time,
        )
        return [(img, (ts, ts)) for img, ts in frames_with_ts]
    if unit == "clip":
        return load_video_sampled_clips(
            video_path=video_path,
            sample_fps=sample_fps,
            clip_duration=clip_duration,
            clip_frames=clip_frames,
            max_clips=max_units,
            start_time=start_time,
            end_time=end_time,
            compose_mode=compose_mode,
        )
    raise ValueError(f"Unsupported stream unit: {unit!r}. Expected 'frame' or 'clip'.")
