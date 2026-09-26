python -m main.cli.train_method_sft \
  --config configs/selectstream_qwen3vl8b_lvm_sft.yaml \
  --train_jsonl path_to_dataset.jsonl \
  --output_dir outputs/qwen3vl8b_method_sft \
  --epochs 1 \
  --default_sample_fps 1.0 \
  --stream_update_unit frame
