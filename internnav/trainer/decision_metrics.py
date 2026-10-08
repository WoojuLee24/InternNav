"""Teacher-forced System2 decision metrics from a training/validation forward pass (no extra forward).

A sample's supervised tokens split into assistant spans (label runs that are not IGNORE and not the
S1 TRAJ tokens the collator appends). The first span is the decision (dataset labels, see
.claude/memory/understanding_s2_label_generation.md):
    "↓"          pixel goal follows -> the second span is "x y" (col row, 640x480 look-down image)
    "STOP"       stop (only at the last frame of an episode in the data)
    "↑←→..."     turn
The model's teacher-forced argmax over the same positions is decoded and classified the same way.
Teacher forcing means every token is predicted from the GT prefix: this is decision accuracy
under GT context, not free generation.

Metrics (DecisionAccumulator.summary): type_acc, stop_precision/recall/f1, turn_exact,
coord_err_px / coord_le10 / coord_parse_fail, per-span-type CE (ce_pixel, ce_stop, ce_turn, ce_coord),
counts n_*. Keys with an empty denominator are omitted (never NaN).
"""
import math
import re

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
TRAJ_TOKEN_ID = 151667  # appended by the collator for System1 queries (internvla_n1_lerobot_dataset)
KINDS = ("pixel", "stop", "turn")


def kind(text: str) -> str:
    t = text.strip()
    if t.startswith("STOP"):
        return "stop"
    if t.startswith("↓"):
        return "pixel"
    if t[:1] in ("↑", "←", "→"):
        return "turn"
    if re.match(r"\d", t):
        return "coord"
    return "other"


def parse_xy(text: str):
    nums = re.findall(r"\d+", text)
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


@torch.no_grad()
def sample_decisions(logits: torch.Tensor, labels: torch.Tensor, decode):
    """Per-sample decision dicts (None for samples without supervised tokens).

    logits [B, L, V] (the forward's own output; only supervised positions are upcast),
    labels [B, L]; decode(list[int]) -> str (tokenizer.decode(..., skip_special_tokens=True))."""
    out = []
    shifted = labels[:, 1:]
    valid = (shifted != IGNORE_INDEX) & (shifted != TRAJ_TOKEN_ID)
    for b in range(labels.shape[0]):
        pos = valid[b].nonzero().squeeze(1)
        if pos.numel() == 0:
            out.append(None)
            continue
        lg = logits[b, pos].float()           # logits at i predict labels at i+1
        tgt = shifted[b, pos]
        ce = F.cross_entropy(lg, tgt, reduction="none")
        pred = lg.argmax(-1)
        cuts = (torch.nonzero(pos[1:] - pos[:-1] != 1).squeeze(1) + 1).tolist()
        bounds = [0] + cuts + [pos.numel()]
        spans = []
        for s, e in zip(bounds, bounds[1:]):
            spans.append({"gt": decode(tgt[s:e].tolist()).strip(), "pred": decode(pred[s:e].tolist()).strip(),
                          "ce": float(ce[s:e].mean())})
        gt_kind, pred_kind = kind(spans[0]["gt"]), kind(spans[0]["pred"])
        d = {"gt": gt_kind, "pred": pred_kind, "gt_text": spans[0]["gt"], "pred_text": spans[0]["pred"],
             "ce": {gt_kind: spans[0]["ce"]}}
        if gt_kind == "turn":
            d["turn_exact"] = spans[0]["pred"] == spans[0]["gt"]
        if gt_kind == "pixel" and len(spans) > 1:
            g, p = parse_xy(spans[1]["gt"]), parse_xy(spans[1]["pred"])
            d["ce"]["coord"] = spans[1]["ce"]
            d["coord_gt"], d["coord_pred"] = g, p
            d["coord_err"] = math.dist(g, p) if (g and p) else None
        out.append(d)
    return out


class DecisionAccumulator:
    """Counters that all-reduce as one tensor (fixed field order)."""

    FIELDS = (["n_" + k for k in KINDS] + ["correct_" + k for k in KINDS]
              + ["stop_tp", "stop_fp", "turn_exact", "coord_n", "coord_err_sum", "coord_le10", "coord_parse_fail"]
              + [f"ce_sum_{k}" for k in KINDS + ("coord",)] + [f"ce_n_{k}" for k in KINDS + ("coord",)])

    def __init__(self):
        self.c = dict.fromkeys(self.FIELDS, 0.0)

    def add(self, d):
        if d is None or d["gt"] not in KINDS:
            return
        c, g = self.c, d["gt"]
        c["n_" + g] += 1
        c["correct_" + g] += d["pred"] == g
        c["stop_tp"] += g == "stop" and d["pred"] == "stop"
        c["stop_fp"] += g != "stop" and d["pred"] == "stop"
        c["turn_exact"] += bool(d.get("turn_exact"))
        if "coord_gt" in d:
            c["coord_n"] += 1
            if d["coord_err"] is None:
                c["coord_parse_fail"] += 1
            else:
                c["coord_err_sum"] += d["coord_err"]
                c["coord_le10"] += d["coord_err"] <= 10
        for k, v in d["ce"].items():
            if k in KINDS + ("coord",):
                c[f"ce_sum_{k}"] += v
                c[f"ce_n_{k}"] += 1

    def tensor(self, device):
        return torch.tensor([self.c[f] for f in self.FIELDS], dtype=torch.float64, device=device)

    def load(self, t):
        self.c = dict(zip(self.FIELDS, t.tolist()))

    def empty(self):
        return sum(self.c["n_" + k] for k in KINDS) == 0

    def summary(self):
        c = self.c
        out = {}
        n = sum(c["n_" + k] for k in KINDS)
        if n:
            out["type_acc"] = sum(c["correct_" + k] for k in KINDS) / n
        for k in KINDS:
            if c["n_" + k]:
                out["n_" + k] = int(c["n_" + k])
                out[f"acc_{k}"] = c["correct_" + k] / c["n_" + k]
        if c["stop_tp"] + c["stop_fp"]:
            out["stop_precision"] = c["stop_tp"] / (c["stop_tp"] + c["stop_fp"])
        if c["n_stop"]:
            out["stop_recall"] = c["stop_tp"] / c["n_stop"]
        if "stop_precision" in out and "stop_recall" in out and out["stop_precision"] + out["stop_recall"]:
            p, r = out["stop_precision"], out["stop_recall"]
            out["stop_f1"] = 2 * p * r / (p + r)
        if c["n_turn"]:
            out["turn_exact"] = c["turn_exact"] / c["n_turn"]
        if c["coord_n"]:
            ok = c["coord_n"] - c["coord_parse_fail"]
            out["coord_parse_fail"] = c["coord_parse_fail"] / c["coord_n"]
            if ok:
                out["coord_err_px"] = c["coord_err_sum"] / ok
                out["coord_le10"] = c["coord_le10"] / ok
        for k in KINDS + ("coord",):
            if c[f"ce_n_{k}"]:
                out[f"ce_{k}"] = c[f"ce_sum_{k}"] / c[f"ce_n_{k}"]
        return out


if __name__ == "__main__":
    # self-check with a toy vocabulary: 0 pad, 1 "↓", 2 "STOP", 3 "←", 4 "→", 5..14 digits, 15 " ", 16 "<im_end>"
    vocab = {1: "↓", 2: "STOP", 3: "←", 4: "→", 15: " ", 16: ""}
    vocab.update({5 + i: str(i) for i in range(10)})

    def decode(ids):
        return "".join(vocab.get(i, "?") for i in ids)

    def make(label_seq, pred_seq, L=24):
        """label_seq: list of (position, token); pred_seq: argmax token for the same positions."""
        labels = torch.full((L,), IGNORE_INDEX)
        logits = torch.zeros(L, 20)
        for (p, tok), pt in zip(label_seq, pred_seq):
            labels[p] = tok
            logits[p - 1, pt] = 5.0
        return labels, logits

    # pixel sample: span "↓" at 3, then "4 12" at 10..13 (digits 4 -> id 9, 1 -> 6, 2 -> 7); pred "4 15"
    l1, g1 = make([(3, 1), (10, 9), (11, 15), (12, 6), (13, 7)], [1, 9, 15, 6, 10])
    # stop sample predicted as pixel
    l2, g2 = make([(5, 2)], [1])
    # turn "←←" predicted exactly, then a TRAJ token that must be ignored
    l3, g3 = make([(4, 3), (5, 3)], [3, 3])
    l3[20] = TRAJ_TOKEN_ID
    labels, logits = torch.stack([l1, l2, l3]), torch.stack([g1, g2, g3])
    ds = sample_decisions(logits, labels, decode)
    assert ds[0]["gt"] == "pixel" and ds[0]["coord_gt"] == (4, 12) and ds[0]["coord_pred"] == (4, 15)
    assert ds[0]["coord_err"] == 3.0 and ds[1]["gt"] == "stop" and ds[1]["pred"] == "pixel"
    assert ds[2]["turn_exact"] and ds[2]["gt_text"] == "←←"
    acc = DecisionAccumulator()
    for d in ds:
        acc.add(d)
    s = acc.summary()
    assert s["type_acc"] == 2 / 3 and s["stop_recall"] == 0.0 and "stop_precision" not in s
    assert s["turn_exact"] == 1.0 and s["coord_err_px"] == 3.0 and s["coord_le10"] == 1.0
    acc2 = DecisionAccumulator()
    acc2.load(acc.tensor("cpu") * 2)  # all-reduce of two identical ranks
    assert acc2.summary()["type_acc"] == s["type_acc"] and acc2.summary()["n_stop"] == 2
    print("decision_metrics self-check OK", {k: round(v, 3) for k, v in s.items()})
