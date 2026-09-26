# SelectStream

Official implementation of **What Should a Streaming Video Model Remember?** (NeurIPS 2026).

[[Paper]](https://arxiv.org/abs/2606.16353) [[Project Page]](https://haonan-ge.github.io/selectstream/)

SelectStream formulates streaming memory as *budgeted online latent evidence allocation*. The current observation stays directly visible to a frozen VLM, while history is exposed only through a compact, query-conditioned evidence budget. Three coordinated mechanisms decide when to write, what to preserve, and how to read:

- **Surprise-driven Adaptive Windowing (SAW)** — closes a segment on a surprise spike, accumulated surprise energy, or the maximum segment length.
- **Latent Visual Memory (LVM)** — writes projected VLM visual embeddings into a fixed-capacity memory graph with gated updates and priority-preserving consolidation.
- **Graph Attention Reasoning (GAR)** — routes the query through a small subgraph, refines it with relational graph attention, and injects the top-`M` nodes as calibrated latent evidence tokens.

Supported backbones: `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen3-VL-8B-Instruct`. The backbone is frozen; only the SelectStream modules are trained.

## Environment

```bash
cd SelectStream
conda create -n selectstream python=3.10 -y
conda activate selectstream
python -m pip install --upgrade pip setuptools wheel
pip install "numpy<2"
pip install -r requirements.txt
pip install opencv-python-headless
```

`transformers>=4.57` is required for Qwen3-VL. `opencv-python-headless` is a video decoding fallback.

## Training Data

SelectStream is trained on [Streamo-Instruct-465K](https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K) ([Streamo](https://github.com/maifoundations/Streamo), Xia et al.), which covers narration, action and event captioning, event grounding, and time-sensitive QA.

1. **Annotations.** Accept the dataset terms on Hugging Face, then download:

   ```bash
   huggingface-cli login
   huggingface-cli download maifoundations/Streamo-Instruct-465K --repo-type dataset --local-dir data/Streamo-Instruct-465K
   ```

2. **Videos.** Videos are not redistributed. Obtain them from the source datasets (ActivityNet, COIN, DiDeMo, Ego-TimeQA, HowTo, Koala, LLaVA-Video, QVHighlights, QuerYD, TACoS, YouCook2) and place them under one root so that `<video_root>/<video_path>` resolves for each annotation.

3. **Convert** to the SelectStream JSONL format:

   ```bash
   python -m main.data.prepare_streamo \
     --anno_dir data/Streamo-Instruct-465K \
     --video_root data/videos \
     --output data/streamo_selectstream.jsonl \
     --skip_missing_videos
   ```

   Each Streamo response becomes one causal training row, following Streamo's time convention (time `t` denotes the second `<t-1 s, t s>`):
   - **Span responses** (`st_time`/`end_time`): the event frames become `evidence_timestamps` for `L_ret`, and the question is answered at the frame right after the event ends, so the evidence lies in the observed history.
   - **Instant responses** (`time`): answered at that frame with the answer loss only.

   Use `--tasks qa event_grounding ...` to keep a subset, or `--max_history_sec` to cap the streamed prefix.

## Data Format

Streaming video QA data is stored as JSONL:

```json
{
  "id": "vid_001_q3",
  "video": "videos/demo.mp4",
  "question": "What happened before this moment?",
  "question_time": 18.5,
  "answer": "He opened the door and entered the room.",
  "evidence_timestamps": [[11.2, 13.0], [14.8, 17.3]],
  "sample_fps": 1.0,
  "stream_update_unit": "frame",
  "start_time": 0.0
}
```

- `video`, `question` (or `prompt`), `question_time`, and `answer` are required for streaming training.
- `evidence_timestamps` (or node ids in `evidence_ids`) enable the retrieval loss `L_ret`; rows without them train with the answer loss only.
- Prompts are wrapped in the backbone chat template automatically.
- Frames are streamed causally at 1 fps: only the prefix up to `question_time` is observed.

## Training

```bash
python -m main.cli.train_method_sft \
  --config configs/selectstream_qwen25vl7b_lvm_sft.yaml \
  --train_jsonl data/streamo_selectstream.jsonl \
  --output_dir outputs/selectstream_qwen25vl7b \
  --epochs 1
```

Use `configs/selectstream_qwen3vl8b_lvm_sft.yaml` for Qwen3-VL-8B. The objective is `L = L_ans + β L_ret + γ L_spar` with AdamW, learning rate `2e-4`, batch size 1, and gradient accumulation 4 (see `training:` in the config).

## Inference

```bash
python -m main.cli.infer_video \
  --config configs/selectstream_qwen25vl7b_lvm_sft.yaml \
  --ckpt outputs/selectstream_qwen25vl7b/epoch0 \
  --video /path/to/video.mp4 \
  --prompt "What happened before this moment?" \
  --show_evidence
```

The answer is generated at the last sampled frame; use `--end_time` or `--query_frame_index` to query earlier moments.

## Evaluation

```bash
python -m main.cli.eval \
  --config configs/selectstream_qwen25vl7b_lvm_sft.yaml \
  --jsonl /path/to/eval.jsonl \
  --ckpt outputs/selectstream_qwen25vl7b/epoch0 \
  --method_infer \
  --report_evidence \
  --metric substr
```

Video rows are evaluated with the causal streaming protocol. With `--report_evidence`, timestamped rows also report `Recall@M` and `T-Overlap` of the injected evidence nodes.

## Default Budgets and Hyperparameters

| Symbol | Meaning | Config key | Default |
| --- | --- | --- | --- |
| `N` | memory slots | `lvm_num_slots` | 256 |
| `B` | retrieved subgraph budget | `method_graph_budget` | 64 |
| `M` | injected evidence tokens | `method_evidence_tokens` | 8 |
| `L_min`, `L_max` | SAW segment length | `lvm_saw_l_min`, `lvm_saw_l_max` | 8, 64 |
| `B_s` | surprise-energy budget | `lvm_saw_energy_budget` | 8.0 |
| `λ`, `ρ` | attention-surprise weight, EMA decay | `lvm_saw_lambda_attn`, `lvm_saw_ema_decay` | 0.5, 0.9 |
| `τ_r`, `τ_s` | update routing thresholds | `lvm_tau_r`, `lvm_tau_s` | 0.75, 0.35 |
| `λ_sim, λ_sup, λ_acc, λ_rec` | consolidation weights | `lvm_merge_*_weight` | 1.0, 0.5, 0.25, 0.25 |
| top-`k`, `K` | seeds, GAR layers | `method_seed_topk`, `method_gar_layers` | 16, 2 |
| `η, ξ_ℓ, ξ_m` | retrieval score terms | `method_eta_surprise_boost`, `method_span_penalty_weight`, `method_merge_penalty_weight` | 0.2, 0.05, 0.05 |
| `α_r`, `κ`, `τ_route` | routing | `method_route_edge_weight`, `method_route_threshold`, `method_route_temperature` | 0.1, 0.0, 1.0 |
| `β`, `γ` | loss weights | `beta_ret`, `gamma_spar` | 0.1, 0.05 |

## Code Structure

| Component | File |
| --- | --- |
| SAW, LVM writing, priority-preserving consolidation | `main/model/lvm_memory.py` |
| Query-conditioned subgraph routing, GAR, evidence calibration | `main/model/dmg_gar.py` |
| Query encoder | `main/model/query_builder.py` |
| Projected visual embeddings, latent evidence injection, decoding | `main/model/model.py` |
| Training objective (`L_ans`, `L_ret`, `L_spar`) and grounding metrics | `main/trainer/method_sft.py` |
| Causal streaming loop | `main/utils/streaming.py` |
| Streamo-Instruct-465K conversion | `main/data/prepare_streamo.py` |

`main/cli/train_stage1.py`, `main/cli/train_stage2.py` and the `vismem_*` configs are legacy compatibility paths and are not part of SelectStream.

## Citation

```bibtex
@inproceedings{ge2026selectstream,
  title     = {What Should a Streaming Video Model Remember?},
  author    = {Ge, Haonan and Wang, Yiwei and Wu, Hang and Cai, Yujun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

Training data comes from [Streamo-Instruct-465K](https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K); please also cite [Streaming Video Instruction Tuning](https://arxiv.org/abs/2512.21334) if you use it.
