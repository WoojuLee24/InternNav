"""InternVLA-N1 training with System2 decision metrics (decision_metrics.py) on train and validation.

Baseline ``internvla_n1_trainer.py`` is NOT modified: same entry-script precedent as
``internvla_n1_bev_provider_trainer.py`` -- swap the module-level ``Trainer`` and wrap
``make_supervised_data_module``, then run the base ``train()``. Selected by
``Params.decision_metrics=True`` (default_config.py); off = the base trainer as before.

- train: teacher-forced decision counters accumulate every step and are all-reduced at each log
  step -> ``train/dec_*`` in wandb (HF prefixes non-eval keys with ``train/``).
- validation (val_ratio > 0): the same counters over the eval pass -> ``eval/dec_*``, plus one JSONL
  row per validation sample in ``<output_dir>/val_metrics/step<global_step>_rank<r>.jsonl``, keyed by
  ``sample_idx`` (index into the full dataset) with scene/episode/frame metadata.
Loss, eval_loss and best-model selection are untouched: metrics come from the logits the forward
already returns (no extra forward).
"""
import json
import os

import torch
from transformers import Trainer

from internnav.trainer.decision_metrics import DecisionAccumulator, sample_decisions


class IndexedSubset(torch.utils.data.Dataset):
    """Subset whose items carry their index into the full dataset (for per-sample val rows)."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        item = dict(self.subset[i])
        item["sample_idx"] = int(self.subset.indices[i])
        return item


class SampleIdxCollator:
    """Moves each instance's sample_idx (-1 for training samples) into batch['sample_idx']."""

    def __init__(self, base):
        self.base = base

    def __call__(self, instances):
        idx = [inst.pop("sample_idx", -1) for inst in instances]
        batch = self.base(instances)
        batch["sample_idx"] = torch.tensor(idx)
        return batch


def sample_meta(combined, idx):
    """scene / episode / frame of CombinedDataset sample `idx` (NavPixelGoalDataset layout); {} otherwise."""
    if not hasattr(combined, "datasets"):
        return {}
    for ds, cum, n in zip(combined.datasets, combined.cum_lengths, combined.lengths):
        if idx < cum:
            j = int(idx - cum + n)
            break
    else:
        return {}
    item = getattr(ds, "list_data_dict", [None] * (j + 1))[j]
    if not isinstance(item, tuple) or len(item) != 10:
        return {}
    ep_id, _, _, height, _, pitch_2, instruction, (start, _end), _, _ = item
    return {"scene_key": ds.scene_keys[j] if hasattr(ds, "scene_keys") else None, "episode": ep_id,
            "frame": int(start), "setting": f"{height}cm_{pitch_2}deg", "instruction": instruction}


class DecisionMetricsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dec = {"train": DecisionAccumulator(), "eval": DecisionAccumulator()}
        self._val_rows = []

    def _decode(self, ids):
        return self.processing_class.decode(ids, skip_special_tokens=True)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if "sample_idx" not in self._signature_columns:  # else RemoveColumnsCollator drops it
            self._signature_columns.append("sample_idx")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_idx = inputs.pop("sample_idx", None)
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        logits = getattr(outputs, "logits", None)
        if labels is not None and logits is not None:
            split = "train" if model.training else "eval"
            for i, d in enumerate(sample_decisions(logits.detach(), labels, self._decode)):
                self._dec[split].add(d)
                if split == "eval" and d is not None and sample_idx is not None and int(sample_idx[i]) >= 0:
                    self._val_rows.append({"sample_idx": int(sample_idx[i]), **d})
        return (loss, outputs) if return_outputs else loss

    def _reduce(self, split):
        acc = self._dec[split]
        t = acc.tensor(self.args.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(t)  # every rank reaches this at the same log/eval step
        acc.load(t)
        self._dec[split] = DecisionAccumulator()
        return acc.summary()

    def log(self, logs, start_time=None):
        if "loss" in logs:  # a training log step (eval logs carry eval_loss, the final log train_loss)
            logs.update({f"dec_{k}": v for k, v in self._reduce("train").items()})
        super().log(logs, start_time)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        self._dec["eval"], self._val_rows = DecisionAccumulator(), []
        out = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        # ponytail: accelerate pads the last eval batch with repeated samples, so the counters can
        # include up to world_size*batch duplicates; the per-sample JSONL is exact (dedupe by sample_idx)
        out.metrics.update({f"{metric_key_prefix}_dec_{k}": v for k, v in self._reduce("eval").items()})
        self._write_val_rows()
        return out

    def _write_val_rows(self):
        ds = self.eval_dataset
        combined = getattr(getattr(ds, "subset", None), "dataset", None)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        d = os.path.join(self.args.output_dir, "val_metrics")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"step{self.state.global_step:06d}_rank{rank}.jsonl"), "w") as f:
            for r in self._val_rows:
                meta = sample_meta(combined, r["sample_idx"]) if combined is not None else {}
                f.write(json.dumps({"step": self.state.global_step, **meta, **r}, ensure_ascii=False) + "\n")
        self._val_rows = []


def main():
    # imported here: the base module needs internnav/trainer on sys.path (true when this file is run
    # as a script, as runner.py does), and the classes above stay importable for tests
    import internnav.trainer.internvla_n1_trainer as _base

    _base.Trainer = DecisionMetricsTrainer
    orig = _base.make_supervised_data_module

    def with_sample_idx(*args, **kwargs):
        dm = orig(*args, **kwargs)
        if dm.get("eval_dataset") is not None:
            dm["eval_dataset"] = IndexedSubset(dm["eval_dataset"])
        dm["data_collator"] = SampleIdxCollator(dm["data_collator"])
        return dm

    _base.make_supervised_data_module = with_sample_idx
    print("[metrics_trainer] System2 decision metrics on (train/dec_*, eval/dec_*, val_metrics/*.jsonl)", flush=True)
    _base.train()


if __name__ == "__main__":
    main()
