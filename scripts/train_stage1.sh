python -m main.cli.train_stage1 \
  --config configs/selectstream_qwen25vl7b_baseline.yaml \
  --train_jsonl path_to_dataset.jsonl \
  --output_dir outputs/stage1 \
  --epochs 1
