"""노은역 ↔ VLN-PE/VLN-CE 포맷 검증 JSON을 시각적 report.html로 만든다."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz_utils import blink_widget_html, image_to_data_uri, save_gallery


ROOT = Path('scripts/dataset_converters/gs_vlnpe/apply_real/format_validation')
DEFAULT_JSON = ROOT / 'noeun_vln_format_validation.json'
VLNPE_RGB = Path(
    'data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/s8pcmisQ38h/'
    'videos/chunk-000/observation.images.rgb/episode_000000.npy'
)
VLNPE_DEPTH = Path(
    'data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/s8pcmisQ38h/'
    'videos/chunk-000/observation.images.depth/episode_000000.npy'
)


def emit(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    return path


def bool_pill(value: bool) -> str:
    label = 'YES' if value else 'NO'
    css = 'good' if value else 'bad'
    return f'<span class="pill {css}">{label}</span>'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation_json', type=Path, default=DEFAULT_JSON)
    args = p.parse_args()
    result = json.loads(args.validation_json.read_text(encoding='utf-8'))
    assets = args.validation_json.parent / 'report_assets'
    assets.mkdir(parents=True, exist_ok=True)

    noeun_root = Path(result['noeun']['root'])
    noeun_rgb = noeun_root / 'episode_000000/rgb/frame_0000.jpg'
    noeun_depth_path = noeun_root / 'episode_000000/depth/frame_0000.png'
    pe_rgb = np.load(VLNPE_RGB, mmap_mode='r')[0]
    pe_depth = np.load(VLNPE_DEPTH, mmap_mode='r')[0]
    Image.fromarray(np.asarray(pe_rgb)).save(assets / 'vlnpe_rgb_frame0.png')

    noeun_raw = np.asarray(Image.open(noeun_depth_path))
    noeun_m = noeun_raw.astype(np.float32) * .001
    noeun_m[(noeun_raw == 0) | (noeun_raw >= 10000)] = np.nan
    pe_m = np.asarray(pe_depth, dtype=np.float32) * 10.0
    pe_m[(pe_depth <= 0) | (pe_depth >= 1.0)] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, image, title in zip(
        axes, [noeun_m, pe_m],
        ['Noeun: uint16 PNG × 0.001 m', 'VLN-PE: float32 NPY × 10 m'],
    ):
        shown = ax.imshow(image, cmap='turbo', vmin=.1, vmax=10)
        ax.set_title(title); ax.axis('off')
    fig.colorbar(shown, ax=axes, label='metric depth [m]', fraction=.025)
    depth_compare = emit(assets / 'depth_metric_comparison.png')

    per_ep = result['noeun']['per_episode']
    ids = [x['episode'] for x in per_ep]
    counts = [x['rgb_frames'] for x in per_ep]
    plt.figure(figsize=(11, 4.5))
    plt.bar(ids, counts, color='#4fd1c5')
    plt.xticks(ids); plt.xlabel('Noeun episode'); plt.ylabel('aligned RGB/depth/pose frames')
    plt.title(f'Noeun 20-episode frame integrity — total {sum(counts):,}')
    frame_chart = emit(assets / 'noeun_episode_frame_counts.png')

    labels = ['Noeun', 'VLN-PE', 'VLN-CE']
    widths = [480, 256, result['vlnce']['rgb_sensor']['width']]
    heights = [270, 256, result['vlnce']['rgb_sensor']['height']]
    x = np.arange(3); width = .35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width/2, widths, width, label='width', color='#7f9cf5')
    plt.bar(x + width/2, heights, width, label='height', color='#e0b34d')
    plt.xticks(x, labels); plt.ylabel('pixels'); plt.title('RGB/depth sensor resolution')
    plt.legend()
    resolution_chart = emit(assets / 'resolution_comparison.png')

    pe = result['compatibility']['noeun_vs_vlnpe']
    ce = result['compatibility']['noeun_vs_vlnce']
    summary = f'''
<div class="stat-row">
 <div class="stat"><div class="label">검사 실행</div><div class="value"><span class="pill good">{result['status']}</span></div></div>
 <div class="stat"><div class="label">노은역 frame</div><div class="value">{result['noeun']['totals']['rgb']:,}</div></div>
 <div class="stat"><div class="label">VLN-PE exact format</div><div class="value">{bool_pill(pe['exact_storage_format_same'])}</div></div>
 <div class="stat"><div class="label">VLN-CE exact format</div><div class="value">{bool_pill(ce['exact_storage_format_same'])}</div></div>
</div>
<p><b>결론:</b> 노은역 04 결과는 RGB·metric depth·camera pose라는 센서 의미는 대응하지만,
VLN-PE 및 VLN-CE와 정확히 같은 저장 포맷이 아니며 기존 loader에 직접 입력할 수 없다.</p>'''

    matrix = [
        ('저장 범주', 'offline frame 폴더', 'offline LeRobot-like episode', 'Habitat online environment'),
        ('RGB', 'frame JPG, 480×270 uint8', 'episode NPY, T×256×256×3 uint8', 'runtime 640×480'),
        ('depth', 'frame PNG uint16 ×0.001m', 'episode NPY float32 ×10m', 'runtime float depth 0~10m'),
        ('pose', 'frame 4×4 NPY', 'parquet position+wxyz quaternion', 'simulator agent/sensor state'),
        ('intrinsic', 'episode 3×3 NPY + config JSON', 'implicit K, parquet에는 없음', 'Habitat HFOV config'),
        ('instruction/action', '없음', 'parquet + tasks/meta', 'gzip episode JSON + action space'),
        ('직접 loader 호환', '기준', 'NO', 'NO'),
    ]
    rows = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in matrix)
    gates = ''.join(
        f'<tr><td>{name}</td><td>{bool_pill(ok)}</td></tr>'
        for name, ok in result['checks'].items()
    )
    pe_requirements = ''.join(f'<li>{x}</li>' for x in result['compatibility']['conversion_required']['for_vlnpe_loader'])
    ce_requirements = ''.join(f'<li>{x}</li>' for x in result['compatibility']['conversion_required']['for_vlnce'])

    body = f'''
<h2>1. 검사 자체의 무결성</h2>
<table><tr><th>검사</th><th>결과</th></tr>{gates}</table>
<p>여기서 PASS는 비교 입력을 정상적으로 파싱하고 검사했다는 뜻이다. 포맷 동일 여부는 별도이며
두 대상 모두 exact format은 NO다.</p>
<h2>2. 실제 RGB 예시</h2>
{blink_widget_html('rgb_compare', [('Noeun Isaac D455', noeun_rgb), ('VLN-PE actual episode', assets / 'vlnpe_rgb_frame0.png')], title='서로 다른 해상도·container의 RGB')}
<h2>3. metric depth로 decode한 뒤 비교</h2>
<img src="{image_to_data_uri(depth_compare)}">
<p>두 depth는 meter로 변환하면 같은 물리량이지만 저장 dtype과 scale은 다르다.</p>
<h2>4. 노은역 20 episode 전수 frame 정렬</h2>
<img src="{image_to_data_uri(frame_chart)}">
<p>각 episode에서 RGB/depth/pose 개수와 frame 번호가 일치하고 총 8,378개다.</p>
<h2>5. 센서 해상도</h2>
<img src="{image_to_data_uri(resolution_chart)}">
<h2>6. 정확한 계약 비교</h2>
<table><tr><th>항목</th><th>노은역</th><th>VLN-PE</th><th>VLN-CE</th></tr>{rows}</table>
<h2>7. VLN-PE loader에 넣으려면</h2><ul>{pe_requirements}</ul>
<h2>8. VLN-CE 환경에 넣으려면</h2><ul>{ce_requirements}</ul>
<h2>9. 판정의 의미</h2>
<p><b>센서 의미 수준:</b> RGB, optical depth, camera pose가 있어 대응한다.</p>
<p><b>파일 수준:</b> container, shape, dtype, scale, pose 표현, episode metadata가 달라 동일하지 않다.</p>
<p><b>학습/평가 수준:</b> 노은역 obs 폴더만으로는 instruction·action·robot state·task metadata가
없으므로 VLN-PE loader나 Habitat VLN-CE dataset에 직접 사용할 수 없다.</p>
'''
    report = save_gallery(
        args.validation_json.parent, 'report.html',
        '노은역 ↔ VLN-PE/VLN-CE 포맷 검증', summary, body,
        eyebrow='gs_vlnpe format validation',
    )
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
