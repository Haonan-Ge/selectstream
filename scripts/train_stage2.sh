python -m main.cli.train_stage2 \
  --config configs/selectstream_qwen25vl7b_baseline.yaml \
  --train_jsonl path_to_dataset.jsonl \
  --init_from outputs/stage1/epoch0 \
  --output_dir outputs/stage2 \
  --epochs 1
