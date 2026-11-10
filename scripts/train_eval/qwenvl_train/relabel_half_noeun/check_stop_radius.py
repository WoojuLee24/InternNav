#!/usr/bin/env python3
"""반경 STOP 라벨 학습 전 검증 (internvla_n1_lerobot_dataset_stop_radius.py).

1. r = 0 / 0.5 / 1.0 / 1.5 별 turn / pixel / stop (고유 개수 = rank0_print 와 같은 집계) + WAIT, 가중치 곱한 총 샘플 수.
2. 경로 길이 구간(0-10 / 10-30 / 30-60 / 60+ m)별 STOP 고유 개수 (마지막 프레임 + 반경).
3. r = 0 의 샘플 목록이 기존 클래스와 튜플 단위로 같은지 (WAIT 셀: NavPixelGoalWaitDataset, 기본: NavPixelGoalDataset).
4. 에피소드 2 개의 반경 STOP 프레임 이미지 + 남은 거리 저장.

데이터 인자는 셀 config 의 실제 학습 argv 에서 DataArguments 로 파싱한다 (학습과 같은 값).

    python3 scripts/train_eval/qwenvl_train/relabel_half_noeun/check_stop_radius.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-both-b1_wait-wait.py --out logs/noeun_stop_radius/check
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "train_eval", "qwenvl_train"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import transformers  # noqa: E402

import internnav.dataset.internvla_n1_lerobot_dataset as D  # noqa: E402
import internnav.dataset.internvla_n1_lerobot_dataset_stop_radius as SR  # noqa: E402
import internnav.dataset.internvla_n1_wait_lerobot_dataset as W  # noqa: E402
from internnav.trainer.internvla_n1_argument import DataArguments  # noqa: E402

RADII = (0.0, 0.5, 1.0, 1.5)


def load_cell(path):
    spec = importlib.util.spec_from_file_location("cell", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cell_data_args(cell_path):
    """셀 PARAMS -> (runner 와 같은 h200 덮어쓰기 + labeled root) -> 학습 argv -> DataArguments."""
    import default_config as dc
    m = load_cell(cell_path)
    from dataclasses import replace
    p = m.PARAMS
    tm = m.TRAIN_MACHINE["h200"]
    p = replace(p, data_root=tm["data_root"], system2_ckpt=tm["system2_ckpt"], nproc_per_node=tm["nproc_per_node"])
    p = replace(p, data_root=dc.labeled_data_root(p.data_root, p.train_label))
    argv = p.train_argv("<out>", "check")
    wait = None
    if "--wait_mode" in argv:
        wait, argv = W.parse_wait_cli_args(argv)
    da, _rest = transformers.HfArgumentParser(DataArguments).parse_args_into_dataclasses(args=argv, return_remaining_strings=True)
    da.model_type = "qwen2.5vl"
    return da, wait, p


def kind(item):
    pose, action = item[9], item[8]
    if pose is not None:
        return "pixel"
    if action == 0:
        return "stop"
    if isinstance(action, list) and action == [W.WAIT_ACTION]:
        return "wait"
    if isinstance(action, list):
        return "turn"
    assert False, f"unreachable item action={action!r}"


def counts(ds, stop_weight, wait_weight):
    c = Counter(kind(it) for it in ds.list_data_dict)
    assert c["stop"] % stop_weight == 0 and c["wait"] % wait_weight == 0, c
    return {"turn": c["turn"], "pixel": c["pixel"], "stop": c["stop"] // stop_weight, "wait": c["wait"] // wait_weight,
            "total": len(ds.list_data_dict)}


def deep_eq(a, b):
    """튜플/리스트/numpy 가 섞인 샘플 비교 (값과 타입 모양까지)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(deep_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def same_samples(x, y):
    return len(x.list_data_dict) == len(y.list_data_dict) and x.scene_keys == y.scene_keys and \
        all(deep_eq(p, q) for p, q in zip(x.list_data_dict, y.list_data_dict))


def build(cls, da, r):
    a = copy.copy(da)
    a.stop_radius_m = r
    return cls(None, a)


def save_episode(out_dir, key, ds_info, actions_poses, r, tag):
    (video, ep_id, instr), (data_path, h, p1, p2, fs, length) = key, ds_info
    actions, poses = actions_poses
    L = len(actions)
    xy = np.asarray(poses)[:, :2, 3]
    d = np.linalg.norm(xy - xy[L - 1], axis=1)
    show = sorted(set(range(max(0, L - 10), L)))   # 끝 10 프레임: 반경 STOP 여부와 남은 거리
    tiles = []
    for f in show:
        img = Image.open(os.path.join(video, f"observation.images.rgb.{h}cm_{p1}deg", f"episode_{ep_id:06d}_{f}.jpg")).convert("RGB")
        img = img.resize((320, 240))
        dr = ImageDraw.Draw(img)
        lab = "STOP(last)" if f == L - 1 else ("STOP(radius)" if f in fs else "-")
        col = (255, 0, 0) if f in fs or f == L - 1 else (255, 255, 255)
        dr.rectangle([0, 0, 320, 38], fill=(0, 0, 0))
        dr.text((4, 2), f"f={f}/{L - 1}  d={d[f]:.2f} m  act={actions[f]}", fill=(255, 255, 255))
        dr.text((4, 20), lab, fill=col)
        if f in fs or f == L - 1:
            dr.rectangle([0, 0, 319, 239], outline=col, width=4)
        tiles.append(img)
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (320 * cols, 240 * rows + 40), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 320, 40 + (i // cols) * 240))
    ImageDraw.Draw(sheet).text((6, 6), f"[{tag}] r={r} m  ep={ep_id}  path={length:.1f} m  radius STOP frames={fs}\n{instr[:180]}",
                               fill=(255, 255, 0))
    scene = video[len(data_path):].lstrip("/").split("/")[0]
    path = os.path.join(out_dir, f"{tag}_{scene}_ep{ep_id}_r{r}.jpg")
    sheet.save(path, quality=90)
    return path, [(f, round(float(d[f]), 3)) for f in fs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="logs/noeun_stop_radius/check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D.local_rank = 0   # 데이터셋 모듈의 rank0_print 를 켠다 (학습에서는 trainer 가 이 전역을 안 바꿔 출력이 없다)

    da, wait, p = cell_data_args(a.config)
    print(f"data_root={da.data_root} vln={da.vln_dataset_use} sample_step={da.sample_step} stop_weight={da.stop_weight} wait={wait}")
    if wait is not None:
        W.set_pending_wait_settings(wait)
        base_cls, sr_cls, wait_w = W.NavPixelGoalWaitDataset, SR.NavPixelGoalWaitStopRadiusDataset, wait["wait_weight"]
    else:
        base_cls, sr_cls, wait_w = D.NavPixelGoalDataset, SR.NavPixelGoalStopRadiusDataset, 1
    report = {"config": a.config, "data_root": da.data_root, "stop_weight": da.stop_weight, "wait": wait}

    # 3. r=0 튜플 비교
    base = base_cls(None, da)
    ds0 = build(sr_cls, da, 0.0)
    same = same_samples(base, ds0)
    report["r0_identical"] = {"same": same, "n_base": len(base.list_data_dict), "n_r0": len(ds0.list_data_dict)}
    print(f"[r=0 vs {base_cls.__name__}] identical={same} ({len(base.list_data_dict)} vs {len(ds0.list_data_dict)})")
    if wait is not None:   # 기본 로더(WAIT 없음)도: 반경 STOP 만 쓰는 클래스 r=0
        b2, s2 = D.NavPixelGoalDataset(None, da), build(SR.NavPixelGoalStopRadiusDataset, da, 0.0)
        same2 = same_samples(b2, s2)
        report["r0_identical_plain"] = {"same": same2, "n": len(b2.list_data_dict)}
        print(f"[r=0 vs NavPixelGoalDataset] identical={same2} ({len(b2.list_data_dict)})")

    # 1·2. 반경별 집계
    info1 = SR.episode_radius_info(da, 1.0)
    ep_len = {k: v[5] for k, v in info1.items()}
    ep_bin = {k: SR.length_bin(v) for k, v in ep_len.items()}
    n_ep_bin = Counter(ep_bin.values())
    rows, bins = [], []
    for r in RADII:
        ds = base if r == 0 else build(sr_cls, da, r)
        c = counts(ds, da.stop_weight, wait_w)
        c["r"] = r
        c["radius_stats"] = getattr(ds, "stop_radius_stats", None)
        rows.append(c)
        per_bin = Counter()
        for it in ds.list_data_dict:
            if kind(it) == "stop":
                per_bin[ep_bin[(it[2], it[0], it[6])]] += 1
        bins.append({"r": r, **{b: per_bin[b] // da.stop_weight for b, _, _ in SR.LENGTH_BINS}})
        print(f"r={r}: {c}")
    report["counts"] = rows
    report["stop_by_length"] = bins
    report["episodes_by_length"] = {b: n_ep_bin[b] for b, _, _ in SR.LENGTH_BINS}

    # 4. 에피소드 2 개 시각화 (r=1.0): 짧은 경로 하나 + 반경 프레임이 가장 많은 것 하나
    ann = {}
    for data in D.data_list(da.vln_dataset_use.split(",")):
        dp = os.path.join(da.data_root, data["data_path"])
        for it in D.get_annotations_from_lerobot_data(dp, f"{data['height']}cm_{data['pitch_2']}deg")["episodes"]:
            ann[(it["video"], it["id"], it["instructions"])] = (it["actions"][1:] + [0], it[f"poses_{data['height']}cm_{data['pitch_2']}deg"])
    cand = [k for k, v in info1.items() if v[4]]
    short = sorted([k for k in cand if ep_len[k] < 10], key=lambda k: ep_len[k])
    most = sorted(cand, key=lambda k: (-len(info1[k][4]), ep_len[k]))
    picks = []
    if short:
        picks.append(("short", short[0]))
    picks.append(("most", next(k for k in most if all(k != q for _, q in picks))))
    vis = []
    for tag, k in picks:
        pth, fd = save_episode(a.out, k, info1[k], ann[k], 1.0, tag)
        vis.append({"tag": tag, "file": pth, "ep": k[1], "path_m": round(ep_len[k], 2), "radius_frames_dist": fd})
        print(f"[vis] {tag}: {pth}  frames(dist)={fd}")
    # 같은 2 에피소드의 r=0.5 / 1.5 프레임
    for tag, k in picks:
        for r in (0.5, 1.5):
            fs = SR.radius_stop_frames(*ann[k], r)
            vis.append({"tag": tag, "ep": k[1], "r": r, "radius_frames": fs})
    report["vis"] = vis
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(f"-> {os.path.join(a.out, 'report.json')}")


if __name__ == "__main__":
    main()
