"""VLNPEMetrics + 같은 층 판정 (다층 씬: 노은역).

VLNPEMetrics 는 NE · success · osr 을 xy 거리로만 잰다 -> 목표 바로 위·아래 층에 멈춰도 성공이 된다.
task_settings['success_dz'] (m) 가 있으면 성공·osr 에 |z - goal_z| < success_dz 를 더한다.
globalgps 와 reference_path 는 둘 다 robot_offset 이 더해진 높이라 z 차이는 그대로 바닥 높이 차다.
기존 xy 판정 값은 `success_xy` · `osr_xy` 로 같이 남긴다 (비교용). success_dz 가 없으면 VLNPEMetrics 와 같다.
"""
from ..configs.metrics.vln_pe_metrics import VLNPEMetricCfg
from .vln_pe_metrics import VLNPEMetrics


@VLNPEMetrics.register('NoeunVLNPEMetric')
class NoeunVLNPEMetrics(VLNPEMetrics):
    def __init__(self, config: VLNPEMetricCfg, task_config):
        super().__init__(config, task_config)
        self.success_dz = getattr(task_config, 'success_dz', None)
        self.dz = None
        self.min_ne_same_floor = float('inf')

    def update(self, task_obs: dict):
        super().update(task_obs)
        obs = task_obs[list(task_obs.keys())[0]]
        if obs['finish_action']:
            self.dz = float(abs(obs['globalgps'][2] - self.goal_position[2]))
            if self.success_dz is None or self.dz < self.success_dz:
                self.min_ne_same_floor = min(self.min_ne_same_floor, self.ne)

    def calc(self):
        metrics = super().calc()[0]
        metrics['success_xy'] = metrics['success']
        metrics['osr_xy'] = metrics['osr']
        metrics['dz'] = self.dz
        if self.success_dz is not None:
            same_floor = self.dz is not None and self.dz < self.success_dz
            metrics['success'] = float(metrics['success'] > 0 and same_floor)
            metrics['osr'] = float(self.min_ne_same_floor < self.success_distance)
            metrics['spl'] = (
                metrics['success'] * self.shortest_path_length / max(self.current_path_length, self.shortest_path_length)
                if self.current_path_length > 0
                else 0
            )
        self.metrics = metrics
        return [metrics]
