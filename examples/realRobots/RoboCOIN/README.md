# unified_80 pretrain (StarVLA)

Train on `/mnt/r/DATA/pre_train_v1/unified_80` with **Qwen3.5 + QwenGR00T**.

## Highlights
- `framework.name: QwenGR00T`, `base_vlm` path containing `Qwen3.5`
- `action_dim / state_dim = 80`
- `include_action_mask: true` — flow-matching loss uses `action_dim_mask`
- `include_state: true` — GR00T state encoder

## Run
```bash
bash examples/realRobots/RoboCOIN/train_files/run_robocoin_train.sh
```

Requires `transformers >= 5.2` for Qwen3.5.
