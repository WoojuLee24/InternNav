# train & eval 260818

```bash
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine {5090, h200} --debug-dir {log_dir}
```

## train

```bash
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine {5090, h200} --no-eval --debug-dir {log_dir}
```

### for debugging

- `--max-steps 2`
- `--debugpy trainer`

## eval (habitat)

```bash
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine {5090, h200} --no-train --debug-dir {log_dir}
```
- [v] 기존 checkpoint 성능 reproduce 확인
```bash
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_full_ld30.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/InternVLA-N1-DualVLN --debug-dir /home/irteam/data-vol2/checkpoints/InternVLA-N1-DualVLN
```

### for debugging: disables use_wandb

- `--max-steps 2 ` 
- `--debugpy eval  `
- `--debug-dir # visualizes bev_v0.1 and input_v0.1 images `

## eval (isaac-sim)

```bash
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine h1 --no-train --debug-dir {log_dir} --model-path {}

# example
python scripts/train_eval/qwenvl_train/runner.py  --config scripts/train_eval/qwenvl_train/image-base/base_s1.fpv_s2.fpv.py --machine h1 --no-train --model-path checkpoints/InternVLA-N1-DualVLN --headless --flash-collision none

```


### additional arguments
- `--debug_dir` 
- `--headless`
- `--flash-collision {"stop", "reset", "none"}`

### for debugging

- `--max-steps 2`
- `--debugpy eval`
- `WANDB_MODE=disabled`

### example

```bash
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/s1.fpv_s2.depth.normalized.concat.py --machine h1 --no-train --max-steps 2 --debug-dir logs/input_v0.1/image_base/s1.fpv_s2.depth.normalized.concat --debugpy eval
```