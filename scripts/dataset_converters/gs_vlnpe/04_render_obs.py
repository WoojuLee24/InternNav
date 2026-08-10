"""M1.4 — 씬 mesh를 렌더링해 rgb/depth/extrinsic을 만든다.

    00: vln_n1 데이터 -> "어떻게 읽고 쓰나"  -> target_schema.json      (전체에 1개)
    01: 씬 mesh/USD    -> "이 씬 써도 되나"    -> scene_meta/<scene>.json (씬마다 1개)
    02: 씬 mesh        -> "어디로 갈 수 있나"  -> esdf/<scene>.npz        (씬마다 1개)
    03: occupancy      -> "어떤 경로로 가나"   -> paths/<scene>.json      (씬마다 1개)
    04: 경로            -> "그 경로에서 뭐가 보이나" -> obs/<scene>/episode_%06d/  (에피소드마다)

## 두 단계 (`--mode`)

렌더러(mesh+intrinsic+pose -> rgb/depth)를 검증 없이 바로 새 경로에 쓰면, 결과가 이상해도
"경로가 이상한지 렌더러가 이상한지"를 못 가른다. 그래서 먼저 실제 GT 카메라 궤적을 그대로 따라가며
렌더링해서 vln_n1의 진짜 캡처 이미지와 직접 비교하고(2-A), 그게 통과해야 03이 만든 경로(reproduce
/random)에 적용한다(2-B). 핵심 렌더 함수 `render_along`은 포즈 소스가 GT든 생성 경로든 동일하다.

  - `gt_replay` (2-A, **기본**) — GT parquet의 실제 `action` 시퀀스(재계획 아님, 원본 그대로)로
    mesh를 렌더링해, `observation.images.rgb/depth`(jpg/png 원본)와 프레임별로 직접 비교한다.
    depth는 median/p95 절대오차[m], rgb는 엣지 상관계수로 "같은 자리에서 같은 구조를 보는가"를 잰다.
  - `reproduce` / `random` (2-B) — 2-A를 통과한 같은 렌더러로, `03_sample_gt_paths.py`가 만든
    `paths/<scene>[_random].json`의 경로를 따라 새 에피소드의 rgb/depth/extrinsic을 만든다.
    카메라 pose는 위치(경로를 프레임 간격으로 리샘플)·yaw(진행방향 tangent)·pitch(그 에피소드의
    `h_b`/`pitch_deg`를 `compose_camera_extrinsic`으로 마운트 회전)를 합성해서 만든다(**신규 로직은
    이 부분뿐** — mesh 로드/렌더러/`action_to_c2w` 컨벤션은 전부 기존 검증된 것을 그대로 쓴다).

## 좌표 컨벤션

렌더러가 받는 `c2w`는 `geometry_utils.action_to_c2w(...)`가 반환하는 것과 **완전히 같은 컨벤션**
(OpenCV: x=right, y=down, z=forward camera-to-world)이다. Open3D의 `Camera.look_at(center, eye, up)`
는 world 벡터만 받으므로, 이 컨벤션 그대로 `eye=c2w[:3,3]`, `forward=c2w[:3,2]`, `up=-c2w[:3,1]`을
넘기면 Open3D 내부 컨벤션(OpenGL, 카메라가 -Z를 본다)과 무관하게 정확히 재현된다 — 새 축 변환을
발명하지 않는다.

## 렌더러 검증 (2-A 스모크 테스트로 이미 확인됨)

씬1 ep0 frame15에서 depth 절대오차 median 0.001 m / p95 0.020 m, rgb는 실제 캡처와 같은 방/가구/
문틀이 같은 자리에 나온다(육안 확인). Open3D `read_triangle_model`(멀티 머티리얼 텍스처 유지,
`read_triangle_mesh` + 단일 `MaterialRecord`는 텍스처가 날아가 회색 평면이 된다 — 실제로 겪음)로
로드하고 `defaultLit`/조명은 `set_indirect_light_intensity` + `enable_sun_light(False)`로 맞춘다.

실행 예시:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04_render_obs.py \\
        --scene 17DRP5sb8fy --mode gt_replay --episodes 0,1,2 --n_frames 6
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import resample_by_arclength  # noqa: E402
from geometry_utils import (  # noqa: E402
    DEPTH_SCALE_RAW_TO_M,
    action_to_c2w,
    colorize_depth,
    compute_mesh_distance,
    find_scene_mesh,
    load_depth_frame_m,
    load_rgb_frame,
    load_scene_mesh,
    save_jpg,
    synthesize_action_poses,
)
from viz_utils import blink_widget_html, save_gallery  # noqa: E402
import dataset_utils  # noqa: E402
from skimage.metrics import structural_similarity as ssim_metric  # noqa: E402

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_n1'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
DEFAULT_SCENE = '17DRP5sb8fy'
SCRIPT_NAME = '04_render_obs'

RENDER_W, RENDER_H = 480, 270          # target_schema.json 실측 (00) — 문서 초안의 480x640은 오기
RENDER_NEAR_M, RENDER_FAR_M = 0.05, 10.0
# 실측(4프레임 MAE 스윕, cx/cy에 ±0.5 조합 6개 vs 기준)으로 확정: Open3D `Camera.set_projection`의
# intrinsics는 픽셀 **중심**이 정수좌표(OpenCV 그대로)가 아니라 **모서리**가 정수좌표인 컨벤션을
# 쓴다 — cx,cy에 +0.5를 더하면 depth MAE가 0.0074 -> 0.00005로 붕괴(다른 조합은 전부 악화).
# 저장용 intrinsic.npy/실측 K는 그대로 두고, **렌더러에 넘길 때만** 이 오프셋을 더한다.
RENDER_PRINCIPAL_POINT_OFFSET_PX = 0.5
# 실측(4프레임 평균 밝기, 인텐시티 스윕)으로 확정: 씬1은 270000에서 render/real 평균 밝기가
# 116.29/116.24로 거의 정확히 일치(기존 90000은 64.8로 실제의 56%밖에 안 됨). 씬2는 같은 값에서
# 124.2/110.3(13% 과다)로 완전히 못 맞춘다 — 실제 카메라는 auto-exposure라 씬마다 다른데 우리
# 조명은 전역 상수 하나뿐이라 원천적으로 두 씬을 동시에 맞출 수 없다(전역 파라미터 한계, 03d에서도
# 반복된 패턴). 90000보다는 훨씬 나은 270000을 쓴다.
#
# 2026-08-05: `ColorGrading` REINHARD 톤매핑(+intensity 310000)으로 std 비율을 1.3x->0.977까지
# 실측상 개선했었지만, 육안 비교에서 사용자가 "처음 버전이 더 낫다"고 판단해 **되돌렸다** — std
# 비율 같은 단일 지표가 좋아져도 전체적인 톤/분위기가 실제로 더 나아 보이는지는 별개 문제였다.
# 이 사건 자체는 `.claude/memory/260804_gs_vlnpe_04_render_obs_result.md`에 기록해 뒀다(교훈:
# 스칼라 지표 최적화가 지각적 개선과 반드시 일치하지 않는다).
INDIRECT_LIGHT_INTENSITY = 270000
MAX_DEPTH_M = 10.0

# --- 2-B(생성 경로 수집) 전용 상수 ---
# 실측 GT 프레임 간 이동거리 median(두 씬 공통, 17DRP5sb8fy/s8pcmisQ38h 모두 0.035 m) — 새로 만든
# 경로도 실제 데이터셋과 같은 공간 밀도로 프레임을 뽑기 위해 이 간격으로 리샘플한다.
FRAME_STEP_M = 0.035
DEPTH_CLIP_MIN_M, DEPTH_CLIP_MAX_M = 0.1, 3.0   # D435i 스펙 (문서 계획)
# 렌더는 mesh에서 직접 만든 depth라 GT의 센서 노이즈가 없다 — mesh-anchor 오차는 GT(2.9e-5 m)보다도
# 작아야 정상이다. 여유를 크게 둔 것은 리샘플/저장(quantize) 과정의 반올림 오차만 흡수하려는 것.
GEN_MESH_ANCHOR_TOL_M = 0.01
GEN_STEP_TOL_RATIO = 3.0   # 인접 프레임 이동거리가 FRAME_STEP_M의 이 배수를 넘으면 이상 신호

# --- 2-A 통과 기준 ---
# `RENDER_PRINCIPAL_POINT_OFFSET_PX` 수정 전 실측(median 0.003 m/p95 0.013~0.014 m)까지는 여유를
# 크게 뒀었다. 수정 후 두 씬 36프레임 전부 median<=0.0001 m, p95<=0.0001 m로 mesh-anchor
# 정밀도(2.9e-5 m)에 근접한다 — 그래도 "같은 mesh를 다시 렌더링" vs "실제 D435i 캡처" 비교라는
# 근본 차이(센서 노이즈·재질 경계 보간) 때문에 mesh-anchor보다는 느슨하게 둔다.
GT_REPLAY_DEPTH_MEDIAN_TOL_M = 0.01
GT_REPLAY_DEPTH_P95_TOL_M = 0.05
# 같은 수정 후 edge_corr median이 두 씬에서 0.43~0.48 -> 0.71~0.74로 뛰었다(픽셀 오프셋이 rgb
# 정합도 같이 갉아먹고 있었다는 뜻) — 프레임별 최솟값 실측(0.622)에 여유를 두고 0.5로 높였다.
GT_REPLAY_RGB_EDGE_CORR_MIN = 0.5


def load_scene_model(mesh_root, scene: str):
    """씬 mesh를 텍스처 포함 `TriangleMeshModel`로 로드.

    `read_triangle_mesh` + 단일 `MaterialRecord`는 멀티 머티리얼(이 씬은 23개 텍스처)을 못 살려
    회색 평면이 되는 것을 실측으로 확인했다 — 반드시 `read_triangle_model`을 쓴다.
    """
    import open3d as o3d
    return o3d.io.read_triangle_model(str(find_scene_mesh(Path(mesh_root), scene)))


def build_renderer(width: int, height: int, k: np.ndarray):
    """`(width,height,K)`로 Open3D `OffscreenRenderer`를 만든다. `render_along`이 매 프레임 재사용.

    `k`는 OpenCV 컨벤션(저장/실측값) 그대로 받는다 — Open3D `set_projection`에 넘기기 **직전에만**
    `RENDER_PRINCIPAL_POINT_OFFSET_PX`를 더한다(실측으로 확정된 픽셀 컨벤션 차이 보정). 호출부가
    저장하는 `k`(intrinsic.npy 등)는 이 함수 밖에서 원본 그대로 쓰면 된다.
    """
    import open3d as o3d
    k_render = np.asarray(k, dtype=np.float64).copy()
    k_render[0, 2] += RENDER_PRINCIPAL_POINT_OFFSET_PX
    k_render[1, 2] += RENDER_PRINCIPAL_POINT_OFFSET_PX
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.camera.set_projection(k_render, RENDER_NEAR_M, RENDER_FAR_M, float(width), float(height))
    renderer.scene.scene.set_indirect_light_intensity(INDIRECT_LIGHT_INTENSITY)
    renderer.scene.scene.enable_sun_light(False)
    return renderer


def set_camera_pose(renderer, c2w: np.ndarray) -> None:
    """`c2w`(4,4, OpenCV camera-to-world — `action_to_c2w`와 같은 컨벤션)로 카메라를 옮긴다."""
    eye = c2w[:3, 3].astype(np.float32)
    forward = c2w[:3, 2].astype(np.float32)   # OpenCV z=forward
    up = -c2w[:3, 1].astype(np.float32)        # OpenCV y=down -> world up = -y열
    renderer.scene.camera.look_at((eye + forward).reshape(3, 1), eye.reshape(3, 1), up.reshape(3, 1))


def render_along(renderer, model, poses_c2w) -> tuple:
    """`(N,4,4)` OpenCV c2w pose -> `rgb`(uint8 (N,H,W,3)), `depth_m`(float32 (N,H,W), 배경=inf).

    포즈 소스가 GT 원본(2-A)이든 03이 만든 경로에서 합성한 것(2-B)이든 완전히 같은 함수를 탄다.
    """
    renderer.scene.clear_geometry()
    renderer.scene.add_model('scene', model)
    rgbs, depths = [], []
    for c2w in poses_c2w:
        set_camera_pose(renderer, np.asarray(c2w, dtype=np.float64))
        rgbs.append(np.asarray(renderer.render_to_image()))
        depths.append(np.asarray(renderer.render_to_depth_image(z_in_view_space=True)))
    renderer.scene.remove_geometry('scene')
    return np.stack(rgbs), np.stack(depths)


def edge_correlation(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """엣지 맵끼리 피어슨 상관계수 — 렌더/실측이 텍스처·조명은 달라도 "같은 자리에서 같은 구조를
    보는가"를 잰다(픽셀값 자체 비교는 GS 렌더와 실제 카메라의 재질/노출 차이 때문에 무의미하다)."""
    ea = cv2.Laplacian(gray_a, cv2.CV_32F, ksize=3)
    eb = cv2.Laplacian(gray_b, cv2.CV_32F, ksize=3)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    denom = float(np.sqrt((ea ** 2).sum() * (eb ** 2).sum()))
    return float((ea * eb).sum() / denom) if denom > 1e-9 else 0.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--dataset', default='vln_n1', choices=['vln_n1', 'vln_pe'],
                        help='GT 데이터셋. vln_pe면 입출력에 _vlnpe 태그, 렌더 해상도 256x256')
    parser.add_argument('--mode', default='gt_replay', choices=['gt_replay', 'reproduce', 'random'],
                        help='gt_replay(2-A, 기본)는 실제 action 시퀀스로 렌더 파이프라인을 검증한다. '
                             'reproduce/random(2-B)은 03이 만든 경로를 따라 새 에피소드를 만든다.')
    parser.add_argument('--episodes', default='0,1,2', help='2-A에서 검증할 에피소드 (콤마 구분)')
    parser.add_argument('--n_frames', type=int, default=6, help='2-A: 에피소드당 균등 표본 프레임 수')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='2-B: paths/<scene>[_random].json에서 렌더링할 에피소드 수(앞에서부터)')
    parser.add_argument('--data_root', default=None, help='생략 시 --dataset의 기본 경로')
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return parser


def validate_gt_replay(args) -> dict:
    """2-A — GT `action` 시퀀스를 그대로 mesh에 렌더링해 실제 캡처와 프레임별로 비교한다."""
    scene_dir = Path(args.data_root) / args.scene
    episodes = [int(e) for e in str(args.episodes).split(',') if e != '']

    k, renderer = None, None
    model = load_scene_model(args.mesh_root, args.scene)

    frames_out, all_depth_err, all_edge_corr, all_ssim = [], [], [], []
    for ep in episodes:
        gt = dataset_utils.load_gt_episode(args.data_root, args.scene, ep, args.dataset)
        if k is None:
            k = gt['k']
            renderer = build_renderer(RENDER_W, RENDER_H, k)
        n = len(gt['poses_c2w'])
        frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
        poses_c2w = [gt['poses_c2w'][f] for f in frame_idx]
        rgb_render, depth_render = render_along(renderer, model, poses_c2w)

        depth_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'
        rgb_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
        for i, frame in enumerate(frame_idx):
            real_depth_m, real_valid = dataset_utils.load_depth_frame_m(depth_dir, ep, frame, MAX_DEPTH_M, args.dataset)
            real_rgb = dataset_utils.load_rgb_frame(rgb_dir, ep, frame, args.dataset)
            render_valid = np.isfinite(depth_render[i])
            both = real_valid & render_valid
            depth_err = np.abs(depth_render[i][both] - real_depth_m[both]) if both.any() else np.array([np.nan])
            corr = edge_correlation(cv2.cvtColor(rgb_render[i], cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(real_rgb, cv2.COLOR_RGB2GRAY))
            ssim_val = float(ssim_metric(real_rgb, rgb_render[i], channel_axis=2, data_range=255))
            all_depth_err.append(depth_err)
            all_edge_corr.append(corr)
            all_ssim.append(ssim_val)
            frames_out.append({
                'episode': ep, 'frame': int(frame), 'coverage_frac': float(both.mean()),
                'depth_median_m': float(np.median(depth_err)), 'depth_p95_m': float(np.percentile(depth_err, 95)),
                'edge_corr': corr, 'ssim': ssim_val, 'rgb_render': rgb_render[i], 'depth_render': depth_render[i],
                'rgb_real': real_rgb, 'depth_real_m': real_depth_m, 'real_valid': real_valid,
            })
            print(f'    ep {ep:>3} frame {frame:>4}: depth median={frames_out[-1]["depth_median_m"]:.4f}m '
                  f'p95={frames_out[-1]["depth_p95_m"]:.4f}m  edge_corr={corr:.3f}  ssim={ssim_val:.3f}  '
                  f'coverage={frames_out[-1]["coverage_frac"]:.3f}')

    all_depth_err = np.concatenate(all_depth_err)
    stats = {
        'n_frames': len(frames_out),
        'depth_median_m': float(np.median(all_depth_err)),
        'depth_p95_m': float(np.percentile(all_depth_err, 95)),
        'edge_corr_median': float(np.median(all_edge_corr)),
        'edge_corr_min': float(np.min(all_edge_corr)),
        'ssim_median': float(np.median(all_ssim)),
        'ssim_min': float(np.min(all_ssim)),
    }
    if args.dataset == 'vln_pe':
        # vln_pe GT는 보행 중 물리 시뮬 캡처라 pose 기록과 depth 렌더 시점이 어긋난다(00의
        # mesh-anchor 실측: 정지 frame0은 6mm지만 보행 프레임은 2~9cm, 오프셋 보정 불가).
        # 렌더러 자체 오차가 아니라 GT의 pose-depth 정합 한계이므로 그만큼 느슨하게 잡는다.
        # p95는 회전 중 물체 경계 시프트 아웃라이어가 지배해(실측 aggregate 1.6m) 회귀 감지력이
        # 없다 — 참고용 상한만 두고, 회귀에 민감한 지표는 median이다(컨벤션이 깨지면 정지
        # 프레임 포함 전 구간이 0.3m+로 뜀 — 00 대조군 실측).
        med_tol, p95_tol = 0.10, 3.0
    else:
        med_tol, p95_tol = GT_REPLAY_DEPTH_MEDIAN_TOL_M, GT_REPLAY_DEPTH_P95_TOL_M
    stats['depth_median_tol_m'], stats['depth_p95_tol_m'] = med_tol, p95_tol
    stats['depth_ok'] = bool(stats['depth_median_m'] <= med_tol
                             and stats['depth_p95_m'] <= p95_tol)
    # RGB 판정은 SSIM(04_render_obs_isaac.py와 동일 기준 — edge_corr는 엣지 픽셀 정합에
    # 과도하게 민감해 참고용으로만 표시. 근거는 isaac 버전 GT_REPLAY_RGB_SSIM_MIN 주석).
    stats['rgb_ok'] = bool(stats['ssim_median'] >= 0.5)
    stats['passed'] = bool(stats['depth_ok'] and stats['rgb_ok'])
    return {'stats': stats, 'frames': frames_out}


def render_gt_replay_report(log_dir: Path, scene: str, result: dict) -> Path:
    stats = result['stats']

    def pill(ok: bool) -> str:
        return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'

    summary = f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(stats["passed"])}</div>
  <div class="stat"><b>depth median</b><span class="pill {"good" if stats["depth_ok"] else "bad"}">
      {stats["depth_median_m"]:.4f} m (기준 {stats["depth_median_tol_m"]} m)</span></div>
  <div class="stat"><b>depth p95</b><span class="pill {"good" if stats["depth_ok"] else "bad"}">
      {stats["depth_p95_m"]:.4f} m (기준 {stats["depth_p95_tol_m"]} m)</span></div>
  <div class="stat"><b>rgb SSIM</b><span class="pill {"good" if stats["rgb_ok"] else "bad"}">
      median {stats["ssim_median"]:.3f} / min {stats["ssim_min"]:.3f} (기준 0.5)</span></div>
  <div class="stat"><b>rgb 엣지 상관(참고용)</b><span class="pill">
      median {stats["edge_corr_median"]:.3f} / min {stats["edge_corr_min"]:.3f}
      </span></div>
</div>
<p>2-A — GT parquet의 실제 <code>action</code> 시퀀스를 그대로 mesh에 렌더링해(재계획 아님, 원본
그대로) <code>observation.images.rgb/depth</code>(실제 캡처)와 프레임별로 비교했다. depth는 유효
픽셀에서의 median/p95 절대오차[m](mesh-anchor보다 강한 검증 — pose·intrinsic·mesh·렌더러 체인
전체가 실제 센서와 일치하는지 직접 잰다), rgb는 엣지맵 상관계수(질감·노출 차이는 있어도 "같은
구조를 같은 자리에서 보는가"). <b>이 검증이 통과해야 2-B(생성 경로 렌더링)로 넘어간다.</b></p>
'''
    body = ''
    for f in result['frames']:
        render_depth_vis = colorize_depth(np.where(np.isfinite(f['depth_render']), f['depth_render'], np.nan),
                                          np.isfinite(f['depth_render']))
        real_depth_vis = colorize_depth(f['depth_real_m'], f['real_valid'])
        frame_dir = log_dir / f'ep{f["episode"]}_frame{f["frame"]}'
        states = [
            ('render rgb', save_jpg(f['rgb_render'], frame_dir / 'render_rgb.jpg')),
            ('real rgb', save_jpg(f['rgb_real'], frame_dir / 'real_rgb.jpg')),
            ('render depth', save_jpg(render_depth_vis, frame_dir / 'render_depth.jpg')),
            ('real depth', save_jpg(real_depth_vis, frame_dir / 'real_depth.jpg')),
        ]
        body += (f'<h4>ep {f["episode"]} frame {f["frame"]} — depth median {f["depth_median_m"]:.4f} m / '
                f'p95 {f["depth_p95_m"]:.4f} m, edge_corr {f["edge_corr"]:.3f}, '
                f'coverage {f["coverage_frac"]:.3f}</h4>'
                + blink_widget_html(f'f{f["episode"]}_{f["frame"]}', states))
    return save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — {scene} (gt_replay 2-A)', summary, body)


def depth_m_to_raw16(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """(H,W) depth[m] -> (H,W) uint16 raw. vln_n1과 같은 컨벤션(raw/10000=m, invalid=0)."""
    raw = np.zeros(depth_m.shape, dtype=np.uint16)
    scaled = np.clip(np.round(depth_m / DEPTH_SCALE_RAW_TO_M), 1, 65534)
    raw[valid_mask] = scaled[valid_mask].astype(np.uint16)
    return raw


def generate_episode(renderer, model, k: np.ndarray, ep_data: dict, ep_dir: Path) -> dict:
    """03의 경로 1개(reproduce/random) -> rgb/depth/extrinsic 프레임 저장.

    2-A로 검증된 `render_along`/`action_to_c2w` 컨벤션을 그대로 쓰고, 새 로직은
    `synthesize_action_poses`(위치+yaw+pitch 합성) 하나뿐이다.
    """
    xy = resample_by_arclength(np.asarray(ep_data['trajectory'], dtype=np.float64), step_m=FRAME_STEP_M)
    action_poses = synthesize_action_poses(xy, ep_data['floor_z'], ep_data['h_b'], ep_data['pitch_deg'])
    poses_c2w = np.stack([action_to_c2w(a, 'cam2world_gl') for a in action_poses])
    rgb, depth_m = render_along(renderer, model, poses_c2w)
    n = len(action_poses)

    (ep_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'depth').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'extrinsic').mkdir(parents=True, exist_ok=True)
    valid_all = np.zeros((n, RENDER_H, RENDER_W), dtype=bool)
    for i in range(n):
        save_jpg(rgb[i], ep_dir / 'rgb' / f'frame_{i:04d}.jpg')
        valid = np.isfinite(depth_m[i]) & (depth_m[i] >= DEPTH_CLIP_MIN_M) & (depth_m[i] <= DEPTH_CLIP_MAX_M)
        valid_all[i] = valid
        cv2.imwrite(str(ep_dir / 'depth' / f'frame_{i:04d}.png'), depth_m_to_raw16(depth_m[i], valid))
        np.save(ep_dir / 'extrinsic' / f'frame_{i:04d}.npy', action_poses[i].astype(np.float32))
    np.save(ep_dir / 'intrinsic.npy', np.asarray(k, dtype=np.float32))

    # --- 검증 (guideline: 개발 목표 관련 코드는 검증 후 저장) ---
    step = np.linalg.norm(np.diff(poses_c2w[:, :2, 3], axis=0), axis=1)
    step_bad = int((step > FRAME_STEP_M * GEN_STEP_TOL_RATIO).sum())

    rng = np.random.RandomState(0)
    anchor_d = []
    for i in range(0, n, max(1, n // 8)):   # 에피소드당 최대 8프레임만 앵커 검증(비용 절감)
        pts_cam = np.stack(np.where(valid_all[i]), axis=1)
        if len(pts_cam) == 0:
            continue
        sel = pts_cam[rng.choice(len(pts_cam), min(300, len(pts_cam)), replace=False)]
        v, u = sel[:, 0], sel[:, 1]
        z = depth_m[i][v, u].astype(np.float64)
        x_cam = (u - k[0, 2]) * z / k[0, 0]
        y_cam = (v - k[1, 2]) * z / k[1, 1]
        pts_cam3 = np.stack([x_cam, y_cam, z], axis=1)
        pts_world = (poses_c2w[i][:3, :3] @ pts_cam3.T).T + poses_c2w[i][:3, 3]
        anchor_d.append(pts_world)
    return {
        'n_frames': n, 'step_max_m': float(step.max()) if n > 1 else 0.0, 'step_bad_n': step_bad,
        'anchor_points_world': np.concatenate(anchor_d, axis=0) if anchor_d else np.empty((0, 3)),
        'rgb_first': rgb[0], 'depth_first': depth_m[0], 'valid_first': valid_all[0],
        'rgb_last': rgb[-1], 'depth_last': depth_m[-1], 'valid_last': valid_all[-1],
    }


def run_generate(args) -> dict:
    """2-B — `paths/<scene>[_random].json`의 경로를 따라 새 에피소드 rgb/depth/extrinsic을 만든다."""
    paths_dir = Path(args.out_dir) / 'paths'
    tag = dataset_utils.dataset_tag(args.dataset)
    json_path = paths_dir / (f'{args.scene}{tag}.json' if args.mode == 'reproduce' else f'{args.scene}{tag}_random.json')
    if not json_path.is_file():
        print(f'  [ERROR] {json_path} 없음 — 03_sample_gt_paths.py --mode {args.mode}를 먼저 돌릴 것')
        return {'error': f'{json_path} not found'}
    paths = json.load(open(json_path))
    episodes_data = paths['episodes'][:args.num_episodes]

    k = dataset_utils.load_gt_episode(args.data_root, args.scene, 0, args.dataset)['k']

    model = load_scene_model(args.mesh_root, args.scene)
    renderer = build_renderer(RENDER_W, RENDER_H, k)
    mesh = load_scene_mesh(args.mesh_root, args.scene)

    obs_dir = Path(args.out_dir) / 'obs' / f'{args.scene}{tag}_{args.mode}'
    results = []
    for i, ep_data in enumerate(episodes_data):
        ep_dir = obs_dir / f'episode_{i:06d}'
        res = generate_episode(renderer, model, k, ep_data, ep_dir)
        anchor_dist = compute_mesh_distance(res['anchor_points_world'], mesh) if len(res['anchor_points_world']) else np.array([np.nan])
        res['anchor_median_m'] = float(np.median(anchor_dist))
        res['episode_id'] = ep_data['episode_id']
        results.append(res)
        print(f'    ep {i:>3} (src episode_id={ep_data["episode_id"]}): {res["n_frames"]} frames, '
              f'step max={res["step_max_m"]:.4f}m (기준 {FRAME_STEP_M * GEN_STEP_TOL_RATIO:.4f}m, '
              f'위반 {res["step_bad_n"]}건), mesh-anchor median={res["anchor_median_m"]:.5f}m -> {ep_dir}')

    stats = {
        'n_episodes': len(results),
        'step_bad_total': int(sum(r['step_bad_n'] for r in results)),
        'anchor_median_m': float(np.median([r['anchor_median_m'] for r in results])) if results else float('nan'),
        'anchor_worst_m': float(max(r['anchor_median_m'] for r in results)) if results else float('nan'),
    }
    stats['step_ok'] = bool(stats['step_bad_total'] == 0)
    stats['anchor_ok'] = bool(stats['anchor_worst_m'] <= GEN_MESH_ANCHOR_TOL_M)
    stats['passed'] = bool(stats['step_ok'] and stats['anchor_ok'] and len(results) == len(episodes_data))
    return {'stats': stats, 'episodes': results, 'obs_dir': obs_dir}


def render_generate_report(log_dir: Path, scene: str, mode: str, result: dict) -> Path:
    stats = result['stats']

    def pill(ok: bool) -> str:
        return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'

    summary = f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(stats["passed"])}</div>
  <div class="stat"><b>에피소드</b><span class="pill">{stats["n_episodes"]}개</span></div>
  <div class="stat"><b>인접 프레임 이동거리</b><span class="pill {"good" if stats["step_ok"] else "bad"}">
      위반 {stats["step_bad_total"]}건 (기준 {FRAME_STEP_M * GEN_STEP_TOL_RATIO:.3f} m)</span></div>
  <div class="stat"><b>mesh-anchor</b><span class="pill {"good" if stats["anchor_ok"] else "bad"}">
      median {stats["anchor_median_m"]:.5f} m / worst {stats["anchor_worst_m"]:.5f} m
      (기준 {GEN_MESH_ANCHOR_TOL_M} m)</span></div>
</div>
<p>2-B — 2-A로 검증된 같은 렌더러로 <code>03_sample_gt_paths.py --mode {mode}</code>가 만든 경로를
따라 실제 rgb/depth/extrinsic을 생성했다. 산출물: <code>{result["obs_dir"]}</code>. depth는 mesh를
직접 렌더링한 것이라 GT보다도 노이즈가 없어야 정상이므로, mesh-anchor(depth를 저장한 extrinsic/
intrinsic으로 다시 unproject해 씬 mesh 표면까지 재는 것)는 저장 파이프라인 자체의 자기 검증이다.</p>
'''
    body = ''
    for r in result['episodes']:
        first_depth_vis = colorize_depth(np.where(r['valid_first'], r['depth_first'], np.nan), r['valid_first'])
        last_depth_vis = colorize_depth(np.where(r['valid_last'], r['depth_last'], np.nan), r['valid_last'])
        frame_dir = log_dir / f'ep{r["episode_id"]}'
        states = [
            ('첫 프레임 rgb', save_jpg(r['rgb_first'], frame_dir / 'first_rgb.jpg')),
            ('첫 프레임 depth', save_jpg(first_depth_vis, frame_dir / 'first_depth.jpg')),
            ('마지막 프레임 rgb', save_jpg(r['rgb_last'], frame_dir / 'last_rgb.jpg')),
            ('마지막 프레임 depth', save_jpg(last_depth_vis, frame_dir / 'last_depth.jpg')),
        ]
        body += (f'<h4>episode_id {r["episode_id"]} — {r["n_frames"]} 프레임, '
                f'mesh-anchor median {r["anchor_median_m"]:.5f} m, '
                f'step max {r["step_max_m"]:.4f} m</h4>'
                + blink_widget_html(f'gen{r["episode_id"]}', states))
    return save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — {scene} ({mode} 2-B)', summary, body)


def main() -> int:
    args = build_argparser().parse_args()
    if args.data_root is None:
        args.data_root = dataset_utils.default_data_root(args.dataset)
    # 렌더 해상도는 데이터셋의 GT 해상도를 따른다(vln_n1 480x270, vln_pe 256x256) — vln_n1은
    # 기존 값 그대로라 기본 경로 동작 불변.
    global RENDER_W, RENDER_H
    RENDER_W, RENDER_H = dataset_utils.render_wh(args.dataset)
    print(f'[{SCRIPT_NAME}] scene={args.scene} mode={args.mode}')

    if args.mode == 'gt_replay':
        result = validate_gt_replay(args)
        stats = result['stats']
        log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}'
        report = render_gt_replay_report(log_dir, args.scene, result)
        print(f'  report html -> {report}')
        print(f'  => {"PASS" if stats["passed"] else "FAIL"} '
              f'(depth median {stats["depth_median_m"]:.4f} m / p95 {stats["depth_p95_m"]:.4f} m, '
              f'ssim median {stats["ssim_median"]:.3f}, edge_corr median {stats["edge_corr_median"]:.3f} 참고)')
        return 0 if stats['passed'] else 1

    result = run_generate(args)
    if 'error' in result:
        return 2
    stats = result['stats']
    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}_{args.mode}'
    report = render_generate_report(log_dir, args.scene, args.mode, result)
    print(f'  obs -> {result["obs_dir"]}')
    print(f'  report html -> {report}')
    print(f'  => {"PASS" if stats["passed"] else "FAIL"} '
          f'({stats["n_episodes"]}개 에피소드, mesh-anchor median {stats["anchor_median_m"]:.5f} m)')
    return 0 if stats['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
