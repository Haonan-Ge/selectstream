python -m main.cli.train_stage1 \
  --config configs/selectstream_qwen3vl8b_lvm_sft.yaml \
  --train_jsonl path_to_dataset.jsonl \
  --output_dir outputs/qwen3vl8b_lvm_sft_stage1 \
  --epochs 1
