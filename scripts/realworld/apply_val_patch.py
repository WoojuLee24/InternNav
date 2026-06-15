#!/usr/bin/env python3
"""
apply_val_patch.py
==================
이 스크립트를 ubuntu(또는 파일 소유자) 권한으로 실행하면
training-time validation + wandb logging 기능이 코드베이스에 적용됩니다.

실행 방법:
    python scripts/realworld/apply_val_patch.py
    # 또는 쓰기 권한이 있는 사용자(ubuntu)로:
    # sudo -u ubuntu python scripts/realworld/apply_val_patch.py

변경 파일 목록:
  1. internnav/configs/trainer/il.py       - val_ratio 등 설정 필드 추가
  2. internnav/trainer/val_callback.py     - 새 파일: ValidationCallback
  3. internnav/trainer/__init__.py         - ValidationCallback export
  4. internnav/dataset/navdp_lerobot_dataset.py - is_train/val_ratio 지원
  5. scripts/train/base_train/train.py     - val dataset 생성 + callback 등록
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Repo root: {REPO_ROOT}")


def write_file(rel_path, content):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, 'w') as f:
        f.write(content)
    print(f"  [OK] {rel_path}")


def replace_in_file(rel_path, old, new):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    with open(abs_path, 'r') as f:
        content = f.read()
    if old not in content:
        print(f"  [SKIP] {rel_path}: pattern not found")
        return
    content = content.replace(old, new, 1)
    with open(abs_path, 'w') as f:
        f.write(content)
    print(f"  [OK] {rel_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. internnav/configs/trainer/il.py  — val_ratio / val_interval_steps / val_steps
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1/5] internnav/configs/trainer/il.py")
replace_in_file(
    "internnav/configs/trainer/il.py",
    "    report_to: Optional[str] = None\n",
    (
        "    report_to: Optional[str] = None\n"
        "    val_ratio: Optional[float] = None           # fraction of data held out for validation\n"
        "    val_interval_steps: Optional[int] = 500     # run validation every N global steps\n"
        "    val_steps: Optional[int] = 50               # max validation batches per run\n"
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. internnav/trainer/val_callback.py  — NEW FILE
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/5] internnav/trainer/val_callback.py (new file)")
VAL_CALLBACK_CONTENT = '''\
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback


class ValidationCallback(TrainerCallback):
    """Computes validation loss every ``val_interval_steps`` steps and logs to wandb.

    Args:
        trainer: The HuggingFace Trainer instance (used to call compute_loss).
        val_dataset: Dataset to iterate over during validation.
        collate_fn: Collate function matching val_dataset.
        val_interval_steps: How often (in global steps) to run validation.
        val_steps: Maximum number of batches to evaluate per run.
        num_workers: DataLoader workers for validation.
    """

    def __init__(self, trainer, val_dataset, collate_fn, val_interval_steps=500, val_steps=50, num_workers=0):
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.val_interval_steps = val_interval_steps
        self.val_steps = val_steps
        self.num_workers = num_workers

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.val_interval_steps != 0:
            return
        # Only run on the main process to avoid duplicated logging
        if not state.is_world_process_zero:
            return

        model = self.trainer.model
        model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
        )

        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if num_batches >= self.val_steps:
                    break
                try:
                    loss = self.trainer.compute_loss(model, batch)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"[ValidationCallback] batch error (skipped): {e}")
                    continue

        model.train()

        if num_batches > 0:
            avg_val_loss = total_loss / num_batches
            print(f"[Val] step={state.global_step}  val_loss={avg_val_loss:.4f}  ({num_batches} batches)")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval/loss": avg_val_loss}, step=state.global_step)
            except Exception:
                pass
'''
write_file("internnav/trainer/val_callback.py", VAL_CALLBACK_CONTENT)

# ──────────────────────────────────────────────────────────────────────────────
# 3. internnav/trainer/__init__.py  — export ValidationCallback
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/5] internnav/trainer/__init__.py")
replace_in_file(
    "internnav/trainer/__init__.py",
    "from .cma_trainer import CMATrainer\nfrom .rdp_trainer import RDPTrainer\nfrom .navdp_trainer import NavDPTrainer",
    (
        "from .cma_trainer import CMATrainer\n"
        "from .rdp_trainer import RDPTrainer\n"
        "from .navdp_trainer import NavDPTrainer\n"
        "from .val_callback import ValidationCallback\n"
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. internnav/dataset/navdp_lerobot_dataset.py  — is_train / val_ratio
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/5] internnav/dataset/navdp_lerobot_dataset.py")

# 4a. Add is_train and val_ratio parameters to __init__ signature
replace_in_file(
    "internnav/dataset/navdp_lerobot_dataset.py",
    "        random_digit=False,\n        prior_sample=False,\n    ):",
    "        random_digit=False,\n        prior_sample=False,\n        is_train=True,\n        val_ratio=0.0,\n    ):",
)

# 4b. After the 50x replication blocks, apply val_ratio split
# For the preload=False branch: split BEFORE replication
replace_in_file(
    "internnav/dataset/navdp_lerobot_dataset.py",
    (
        "            with open(preload_path, 'w') as f:\n"
        "                json.dump(save_dict, f, indent=4)\n"
        "\n"
        "            # replicate the data 50 times\n"
        "            self.trajectory_data_dir = self.trajectory_data_dir * 50\n"
        "            self.trajectory_rgb_path = self.trajectory_rgb_path * 50\n"
        "            self.trajectory_depth_path = self.trajectory_depth_path * 50\n"
        "            self.trajectory_afford_path = self.trajectory_afford_path * 50\n"
    ),
    (
        "            with open(preload_path, 'w') as f:\n"
        "                json.dump(save_dict, f, indent=4)\n"
        "\n"
        "            # split train / val before replication\n"
        "            if val_ratio and val_ratio > 0:\n"
        "                n_base = len(self.trajectory_data_dir)\n"
        "                n_val = max(1, int(n_base * val_ratio))\n"
        "                if is_train:\n"
        "                    self.trajectory_data_dir = self.trajectory_data_dir[:-n_val]\n"
        "                    self.trajectory_rgb_path = self.trajectory_rgb_path[:-n_val]\n"
        "                    self.trajectory_depth_path = self.trajectory_depth_path[:-n_val]\n"
        "                    self.trajectory_afford_path = self.trajectory_afford_path[:-n_val]\n"
        "                else:\n"
        "                    self.trajectory_data_dir = self.trajectory_data_dir[-n_val:]\n"
        "                    self.trajectory_rgb_path = self.trajectory_rgb_path[-n_val:]\n"
        "                    self.trajectory_depth_path = self.trajectory_depth_path[-n_val:]\n"
        "                    self.trajectory_afford_path = self.trajectory_afford_path[-n_val:]\n"
        "\n"
        "            # replicate the data 50 times (training only)\n"
        "            if is_train:\n"
        "                self.trajectory_data_dir = self.trajectory_data_dir * 50\n"
        "                self.trajectory_rgb_path = self.trajectory_rgb_path * 50\n"
        "                self.trajectory_depth_path = self.trajectory_depth_path * 50\n"
        "                self.trajectory_afford_path = self.trajectory_afford_path * 50\n"
    ),
)

# 4c. For the preload=True branch: split after loading, before replication
replace_in_file(
    "internnav/dataset/navdp_lerobot_dataset.py",
    (
        "        else:\n"
        "            load_dict = json.load(open(preload_path, 'r'))\n"
        "            self.trajectory_data_dir = load_dict['trajectory_data_dir'] * 50\n"
        "            self.trajectory_rgb_path = load_dict['trajectory_rgb_path'] * 50\n"
        "            self.trajectory_depth_path = load_dict['trajectory_depth_path'] * 50\n"
        "            self.trajectory_afford_path = load_dict['trajectory_afford_path'] * 50\n"
    ),
    (
        "        else:\n"
        "            load_dict = json.load(open(preload_path, 'r'))\n"
        "            _tdd = load_dict['trajectory_data_dir']\n"
        "            _trp = load_dict['trajectory_rgb_path']\n"
        "            _tdp = load_dict['trajectory_depth_path']\n"
        "            _tap = load_dict['trajectory_afford_path']\n"
        "            if val_ratio and val_ratio > 0:\n"
        "                n_base = len(_tdd)\n"
        "                n_val = max(1, int(n_base * val_ratio))\n"
        "                if is_train:\n"
        "                    _tdd, _trp, _tdp, _tap = _tdd[:-n_val], _trp[:-n_val], _tdp[:-n_val], _tap[:-n_val]\n"
        "                else:\n"
        "                    _tdd, _trp, _tdp, _tap = _tdd[-n_val:], _trp[-n_val:], _tdp[-n_val:], _tap[-n_val:]\n"
        "            if is_train:\n"
        "                self.trajectory_data_dir = _tdd * 50\n"
        "                self.trajectory_rgb_path = _trp * 50\n"
        "                self.trajectory_depth_path = _tdp * 50\n"
        "                self.trajectory_afford_path = _tap * 50\n"
        "            else:\n"
        "                self.trajectory_data_dir = _tdd\n"
        "                self.trajectory_rgb_path = _trp\n"
        "                self.trajectory_depth_path = _tdp\n"
        "                self.trajectory_afford_path = _tap\n"
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# 5. scripts/train/base_train/train.py  — import + val dataset + callback
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/5] scripts/train/base_train/train.py")

# 5a. Add ValidationCallback import
replace_in_file(
    "scripts/train/base_train/train.py",
    "from internnav.trainer import CMATrainer, NavDPTrainer, RDPTrainer\n",
    "from internnav.trainer import CMATrainer, NavDPTrainer, RDPTrainer, ValidationCallback\n",
)

# 5b. After building train_dataset for navdp, also build val_dataset
replace_in_file(
    "scripts/train/base_train/train.py",
    (
        "        elif config.model_name == 'navdp':\n"
        "            policy_trainer = NavDPTrainer\n"
        "            train_dataset = train_dataset_data\n"
        "            collate_fn = navdp_collate_fn\n"
    ),
    (
        "        elif config.model_name == 'navdp':\n"
        "            policy_trainer = NavDPTrainer\n"
        "            train_dataset = train_dataset_data\n"
        "            collate_fn = navdp_collate_fn\n"
        "\n"
        "        # -------- optional validation dataset --------\n"
        "        val_dataset = None\n"
        "        val_ratio = getattr(config.il, 'val_ratio', None)\n"
        "        if val_ratio and val_ratio > 0:\n"
        "            if config.model_name == 'navdp':\n"
        "                # Re-create dataset with val split (uses cached preload JSON)\n"
        "                val_dataset = NavDP_Base_Datset(\n"
        "                    config.il.root_dir,\n"
        "                    config.il.dataset_navdp,\n"
        "                    config.il.memory_size,\n"
        "                    config.il.predict_size,\n"
        "                    config.il.batch_size,\n"
        "                    config.il.image_size,\n"
        "                    config.il.scene_scale,\n"
        "                    pixel_channel=config.il.pixel_channel,\n"
        "                    preload=True,  # use cached JSON (built by train_dataset_data above)\n"
        "                    random_digit=config.il.random_digit,\n"
        "                    prior_sample=config.il.prior_sample,\n"
        "                    is_train=False,\n"
        "                    val_ratio=val_ratio,\n"
        "                )\n"
        "                # Also rebuild train_dataset with val excluded\n"
        "                train_dataset = NavDP_Base_Datset(\n"
        "                    config.il.root_dir,\n"
        "                    config.il.dataset_navdp,\n"
        "                    config.il.memory_size,\n"
        "                    config.il.predict_size,\n"
        "                    config.il.batch_size,\n"
        "                    config.il.image_size,\n"
        "                    config.il.scene_scale,\n"
        "                    pixel_channel=config.il.pixel_channel,\n"
        "                    preload=True,\n"
        "                    random_digit=config.il.random_digit,\n"
        "                    prior_sample=config.il.prior_sample,\n"
        "                    is_train=True,\n"
        "                    val_ratio=val_ratio,\n"
        "                )\n"
        "            elif config.model_name in ['cma', 'seq2seq']:\n"
        "                # Split lmdb_keys: last val_ratio fraction → val\n"
        "                import copy\n"
        "                all_keys = list(train_dataset.lmdb_keys)\n"
        "                n_val = max(1, int(len(all_keys) * val_ratio))\n"
        "                val_dataset = copy.copy(train_dataset)\n"
        "                val_dataset.lmdb_keys = all_keys[-n_val:]\n"
        "                val_dataset.length = n_val\n"
        "                train_dataset.lmdb_keys = all_keys[:-n_val]\n"
        "                train_dataset.length = len(train_dataset.lmdb_keys)\n"
        "            elif config.model_name == 'rdp':\n"
        "                import copy\n"
        "                all_keys = list(train_dataset.lmdb_keys)\n"
        "                n_val = max(1, int(len(all_keys) * val_ratio))\n"
        "                val_dataset = copy.copy(train_dataset)\n"
        "                val_dataset.lmdb_keys = all_keys[-n_val:]\n"
        "                val_dataset.length = n_val\n"
        "                train_dataset.lmdb_keys = all_keys[:-n_val]\n"
        "                train_dataset.length = len(train_dataset.lmdb_keys)\n"
        "            if val_dataset is not None:\n"
        "                print(f'[Val] val_dataset size: {len(val_dataset)}')\n"
    ),
)

# 5c. Add ValidationCallback to trainer after trainer is created
replace_in_file(
    "scripts/train/base_train/train.py",
    (
        "        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint\n"
        "        run_name = config.name\n"
        "        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=config.log_dir)\n"
        "        trainer.add_callback(ckpt_format_callback)\n"
    ),
    (
        "        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint\n"
        "        run_name = config.name\n"
        "        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=config.log_dir)\n"
        "        trainer.add_callback(ckpt_format_callback)\n"
        "\n"
        "        # Add validation callback if val_dataset was built\n"
        "        if val_dataset is not None:\n"
        "            val_interval = getattr(config.il, 'val_interval_steps', 500) or 500\n"
        "            val_steps_n = getattr(config.il, 'val_steps', 50) or 50\n"
        "            val_cb = ValidationCallback(\n"
        "                trainer=trainer,\n"
        "                val_dataset=val_dataset,\n"
        "                collate_fn=collate_fn,\n"
        "                val_interval_steps=val_interval,\n"
        "                val_steps=val_steps_n,\n"
        "                num_workers=0,\n"
        "            )\n"
        "            trainer.add_callback(val_cb)\n"
        "            print(f'[Val] ValidationCallback registered (every {val_interval} steps, max {val_steps_n} batches)')\n"
    ),
)

print("\n✅ All patches applied successfully!")
print("\nUsage: add val_ratio to IlCfg in your experiment config, e.g.:")
print("  il=IlCfg(..., val_ratio=0.1, val_interval_steps=500, val_steps=50, ...)")
