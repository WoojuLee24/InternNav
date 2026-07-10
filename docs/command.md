## progress (260710)
Done (train & eval & h1): s1.bev.concat, s1.{bev, depth}.replace, s2.{bev, depth}.concat
not yet: s1.depth.concat, s2.{bev, depth}.replace

## train & eval
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine {5090, h200} --debug-dir {log_dir}

## train
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine {5090, h200} --no-eval --debug-dir {log_dir}
### for debugging
--max-steps 2 
--debugpy trainer

## eval (habitat)
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine {5090, h200} --no-train --debug-dir {log_dir}
### for debugging
--max-steps 2 
--debugpy eval

## eval (isaac-sim)
python scripts/train_eval/qwenvl_train/runner.py --config {config.py} --machine h1 --no-train --debug-dir {log_dir}
### for debugging
--max-steps 2 
--debugpy eval

## example
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/s1.fpv_s2.depth.normalized.concat.py --machine h1 --no-train  --max-steps 2   --debug-dir logs/input_v0.1/image_base/s1.fpv_s2.depth.normalized.concat --debugpy eval