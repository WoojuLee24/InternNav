import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback


# TODO: remove after verification
def _save_val_batch_images(batch, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ig = batch["batch_ig"][0].cpu().numpy()  # (H, W, 6)
    current = (ig[:, :, 3:6] * 255).astype("uint8")
    goal    = (ig[:, :, 0:3] * 255).astype("uint8")
    cv2.imwrite(f"{save_dir}/step{step:06d}_current.png", cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_dir}/step{step:06d}_goal.png",    cv2.cvtColor(goal,    cv2.COLOR_RGB2BGR))


class ValidationCallback(TrainerCallback):
    """Computes validation loss and logs to wandb.

    Mode is determined automatically:
      - val_interval_steps is set (> 0) → step mode: runs every N steps
      - val_interval_steps is None       → epoch mode: runs every epoch

    Args:
        val_interval_steps: interval in steps (None = epoch mode).
        debug: if True, saves first batch images to log_dir.
    """

    def __init__(self, trainer, val_dataset, collate_fn, val_interval_steps=None,
                 num_workers=0, debug=False, log_dir="val_batch_debug"):
        self.trainer = trainer
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.val_interval_steps = val_interval_steps  # None → epoch mode
        self.num_workers = num_workers
        self.debug = debug
        self.log_dir = log_dir

    def _validate(self, args, state):
        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        rng_state = np.random.get_state()
        np.random.seed(42)

        model = self.trainer.model
        model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
        )

        device = next(model.parameters()).device
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    if self.debug and num_batches == 0 and rank == 0:
                        _save_val_batch_images(batch, state.global_step, self.log_dir)
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    loss = self.trainer.compute_loss(model, batch)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"[ValidationCallback] batch error (skipped): {e}")
                    continue

        model.train()
        np.random.set_state(rng_state)

        # Aggregate loss across all ranks
        if is_dist:
            stats = torch.tensor([total_loss, float(num_batches)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats[0].item(), int(stats[1].item())

        if rank == 0 and num_batches > 0:
            avg_val_loss = total_loss / num_batches
            print(f"[Val] step={state.global_step} epoch={int(state.epoch)}  val_loss={avg_val_loss:.4f}  ({num_batches} batches)")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"val/loss": avg_val_loss, "val/epoch": state.epoch}, step=state.global_step)
            except Exception:
                pass

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.val_interval_steps is not None:
            return  # step mode handles validation
        self._validate(args, state)

    def on_step_end(self, args, state, control, **kwargs):
        if self.val_interval_steps is None:
            return  # epoch mode handles validation
        if state.global_step % self.val_interval_steps != 0:
            return
        self._validate(args, state)
