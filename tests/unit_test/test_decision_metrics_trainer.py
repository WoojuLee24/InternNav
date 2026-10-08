"""DecisionMetricsTrainer on CPU with a toy model: train logs get dec_* keys, evaluate() gets
eval_dec_* keys and writes val_metrics JSONL keyed by sample_idx (no GPU, no InternVLA weights).

    pytest tests/unit_test/test_decision_metrics_trainer.py -q
"""
import glob
import json
import os

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

VOCAB = {1: "↓", 2: "STOP", 3: "←", 15: " "}
VOCAB.update({5 + i: str(i) for i in range(10)})


class ToyTok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB.get(int(i), "") for i in ids)


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20, 16)
        self.head = nn.Linear(16, 20)

    def forward(self, input_ids, labels=None):
        logits = self.head(self.emb(input_ids))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 20), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)


class ToyData(torch.utils.data.Dataset):
    """pixel ("↓" then "4 12"), stop and turn samples; input == labels where supervised."""

    SAMPLES = [[(3, 1), (6, 9), (7, 15), (8, 6), (9, 7)], [(4, 2)], [(4, 3), (5, 3)]]

    def __len__(self):
        return 12

    def __getitem__(self, i):
        labels = torch.full((12,), -100)
        for p, t in self.SAMPLES[i % 3]:
            labels[p] = t
        return {"input_ids": labels.clamp(min=0), "labels": labels}


def collate(instances):
    return {k: torch.stack([x[k] for x in instances]) for k in instances[0]}


def test_metrics_trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    from transformers import TrainingArguments

    from internnav.trainer.internvla_n1_metrics_trainer import DecisionMetricsTrainer, IndexedSubset, SampleIdxCollator

    full = ToyData()
    train = torch.utils.data.Subset(full, list(range(9)))
    val = IndexedSubset(torch.utils.data.Subset(full, [9, 10, 11]))
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=3, per_device_eval_batch_size=3,
                             max_steps=4, logging_steps=2, eval_strategy="no", report_to=[], use_cpu=True,
                             save_strategy="no", learning_rate=0.1)
    tr = DecisionMetricsTrainer(model=ToyLM(), args=args, train_dataset=train, eval_dataset=val,
                                data_collator=SampleIdxCollator(collate), processing_class=ToyTok())
    tr.train()
    train_logs = [h for h in tr.state.log_history if "loss" in h]
    assert train_logs and all("dec_type_acc" in h and "dec_n_stop" in h for h in train_logs), train_logs
    m = tr.evaluate()
    assert {"eval_loss", "eval_dec_type_acc", "eval_dec_n_pixel", "eval_dec_ce_coord"} <= set(m), m
    rows = [json.loads(line) for f in glob.glob(os.path.join(tmp_path, "val_metrics", "*.jsonl")) for line in open(f)]
    assert sorted(r["sample_idx"] for r in rows) == [9, 10, 11]
    assert {r["gt"] for r in rows} == {"pixel", "stop", "turn"}
