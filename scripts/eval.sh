python -m main.cli.eval \
  --config configs/selectstream_qwen25vl7b_baseline.yaml \
  --jsonl path_to_dataset.jsonl \
  --ckpt outputs/stage2/epoch0 \
  --enable_vismem \
  --metric substr
