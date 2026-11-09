"""노은역 VLN-PE 평가 태스크 — VLNEvalTask 에 데이터 렌더 조건만 덧붙인다.

데이터 렌더러(`gs_vlnpe/apply_real/04_render_obs_isaac.py`, `nav3d/noeun/51_render_rbq.py`)와 같게:
  - 조명: ambient_only (조명 prim 없음) + `/rtx/sceneDb/ambientLightIntensity` + `/rtx/post/tonemap/filmIso`,
    자동 노출(histogram) 끔. RTX 가 씬 로드 때 값을 덮어쓰므로 씬을 올린 뒤(load / post_reset)에 설정한다.
  - `*_collision.usdz` 의 숨은 collision mesh 를 보이게 한다 (rgb = NuRec GS 볼륨, depth = collision mesh).
    flash 컨트롤러의 바닥 raycast 도 이 mesh 를 쓰므로 정밀 충돌(approximation none)로 둔다.
  - 카메라 hfov (60cm_15deg rig = 68.7 deg) 를 focal length 로 맞춘다.
"""
import math
import os

import numpy as np

from ..configs.tasks.noeun_vln_eval_task import NoeunVLNEvalTaskCfg
from .vln_eval_task import VLNEvalTask


@VLNEvalTask.register('NoeunVLNEvalTask')
class NoeunVLNEvalTask(VLNEvalTask):
    def __init__(self, config, scene):
        super().__init__(config, scene)
        # generate_vln_episode 는 VLNEvalTaskCfg(**task_settings) 로 만든다 (extra 허용) -> 노은역 필드 기본값을 여기서 채운다
        self.ncfg = NoeunVLNEvalTaskCfg(**config.model_dump())

    def create_light(self):
        if self.ncfg.light_mode == 'default':
            return super().create_light()
        elif self.ncfg.light_mode == 'ambient_only':
            return None
        else:
            assert False, f"unreachable light_mode={self.ncfg.light_mode!r}"

    def reset_light_position(self, position):
        if self.ncfg.light_mode == 'default':
            return super().reset_light_position(position)
        elif self.ncfg.light_mode == 'ambient_only':
            return None
        else:
            assert False, f"unreachable light_mode={self.ncfg.light_mode!r}"

    def _apply_render_settings(self):
        import carb

        settings = carb.settings.get_settings()
        settings.set_bool('/rtx/post/histogram/enabled', False)
        if self.ncfg.light_mode == 'ambient_only':
            settings.set_float('/rtx/sceneDb/ambientLightIntensity', float(self.ncfg.rtx_ambient))
        settings.set_float('/rtx/post/tonemap/filmIso', float(self.ncfg.film_iso))

    def _expose_collision_meshes(self):
        """씬(로봇 제외)의 숨은 collision mesh 를 보이게 하고 정밀 충돌로 둔다. 바꾼 mesh 수를 돌려준다."""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        scene_root = f'{self.root_path}/scene'  # InternUtopia IsaacsimScene: World/env_{id}/scene
        changed = 0
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(scene_root):
                continue
            if not prim.IsA(UsdGeom.Mesh) or 'PhysicsCollisionAPI' not in set(prim.GetAppliedSchemas()):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
                imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
            changed += 1
        return changed

    def _apply_camera_intrinsics(self):
        """pano_camera_0 의 hfov 를 config 값으로 (horizontal aperture 는 그대로, focal length 만 바꾼다)."""
        if self.ncfg.camera_hfov_deg is None or 'pano_camera_0' not in self.robot.sensors:
            return
        from pxr import Gf, UsdGeom

        camera = self.robot.sensors['pano_camera_0']._camera
        usd_camera = UsdGeom.Camera(camera.prim.prim)
        h_ap = usd_camera.GetHorizontalApertureAttr().Get()
        w, h = self.robot.sensors['pano_camera_0'].resolution
        focal = h_ap / 2.0 / math.tan(math.radians(self.ncfg.camera_hfov_deg) / 2.0)
        usd_camera.GetFocalLengthAttr().Set(focal)
        usd_camera.GetVerticalApertureAttr().Set(h_ap * h / w)  # 정사각 픽셀
        usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(*self.ncfg.camera_clip))

    def _floor_below(self, position):
        """position 아래 첫 바닥 z (로봇 자신 제외). 없으면 None."""
        from omni.physx import get_physx_scene_query_interface

        robot_root = self.robot.articulation.prim.GetPath().pathString.rsplit('/', 1)[0]
        hits = []

        def report(hit):
            path = hit.rigid_body if hit.rigid_body else hit.collision
            if not str(path).startswith(robot_root):
                hits.append((hit.distance, hit.position[2]))
            return True

        origin = (float(position[0]), float(position[1]), float(position[2]))
        get_physx_scene_query_interface().raycast_all(origin, (0.0, 0.0, -1.0), 3.0, report)
        return min(hits)[1] if hits else None

    def _place_camera_on_floor(self):
        """vln_ce rig 처럼 카메라를 로봇 xy · yaw 에 두고 바닥 + camera_floor_height, 아래 camera_pitch_deg 로 놓는다."""
        pos, quat = self.robot.articulation.get_world_pose()  # quat = (w, x, y, z)
        floor = self._floor_below(pos)
        if floor is None:
            floor = pos[2] - self.ncfg.camera_base_height
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        p = math.radians(self.ncfg.camera_pitch_deg)
        f = np.array([math.cos(yaw) * math.cos(p), math.sin(yaw) * math.cos(p), -math.sin(p)])
        r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        u = np.cross(r, f)
        R = np.column_stack([r, u, -f])  # USD 카메라: +X 오른쪽, +Y 위, -Z 전방
        from scipy.spatial.transform import Rotation

        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        position = np.array([pos[0], pos[1], floor + self.ncfg.camera_floor_height])
        cam = self.robot.sensors['pano_camera_0']
        if os.environ.get('NOEUN_DEBUG_CAMERA') == '1':
            c_pos, c_q = cam.get_world_pose()
            Rc = Rotation.from_quat([c_q[1], c_q[2], c_q[3], c_q[0]]).as_matrix()
            print(f'[NOEUN CAM] base {np.round(pos, 3)} quat {np.round(quat, 4)} yaw {math.degrees(yaw):.1f} floor {floor:.3f} '
                  f'| mount fwd {np.round(-Rc[:, 2], 3)} new fwd {np.round(f, 3)}', flush=True)
        cam.set_world_pose(position, np.array([qw, qx, qy, qz]))

    def load(self):
        super().load()
        n = self._expose_collision_meshes()
        print(f'[NoeunVLNEvalTask] collision mesh {n} 개 보이게 함 · light={self.ncfg.light_mode}', flush=True)
        self._apply_render_settings()

    def _hide_robot(self):
        """로봇의 보이는 geometry 를 렌더에서 숨긴다 (카메라가 바닥 위 0.6 m = H1 다리 사이라 다리가 찍힌다). 물리는 그대로."""
        from pxr import UsdGeom

        root = self.robot.articulation.prim.GetPath().pathString.rsplit('/', 1)[0]
        stage = self.robot.articulation.prim.GetStage()
        n = 0
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(root) and prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).MakeInvisible()
                n += 1
        return n

    def _robot_physics_off(self):
        """로봇 collision 과 중력을 끈다 (flash 전용). 바꾼 prim 수를 돌려준다."""
        from pxr import PhysxSchema, UsdPhysics

        root = self.robot.articulation.prim.GetPath().pathString.rsplit('/', 1)[0]
        stage = self.robot.articulation.prim.GetStage()
        n = 0
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(root):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                n += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
        return n

    def post_reset(self) -> None:
        super().post_reset()
        if self.ncfg.robot_physics == 'on':
            pass
        elif self.ncfg.robot_physics == 'off':
            n = self._robot_physics_off()
            print(f'[NoeunVLNEvalTask] robot collision {n} 개 · 중력 끔 (robot_physics=off)', flush=True)
        else:
            assert False, f"unreachable robot_physics={self.ncfg.robot_physics!r}"
        if self.ncfg.hide_robot:
            self._hide_robot()
        self._apply_camera_intrinsics()
        self._apply_render_settings()

    def get_rgb_depth(self):
        """VLNEvalTask.get_rgb_depth 와 같고 RTX subframe 수만 config 값 (GS 볼륨 수렴)."""
        obs = {}
        if 'pano_camera_0' in self.robot.sensors:
            camera = self.robot.sensors['pano_camera_0']
            import omni.replicator.core as rep

            from internnav.evaluator.utils.common import norm_depth

            if self.ncfg.camera_floor_height is not None:
                self._place_camera_on_floor()
            if self.env_id == 0:
                rep.orchestrator.step(rt_subframes=self.ncfg.render_subframes, delta_time=0.0, pause_timeline=False)
            cur_obs = camera.get_data()
            obs['depth'] = norm_depth(cur_obs['depth'])[..., np.newaxis]
            obs['rgb'] = cur_obs['rgba'][..., :3]
            c_pos, c_ori = camera.get_world_pose()
            obs['camera_position'] = c_pos - self.env_offset
            obs['camera_orientation'] = c_ori

        if 'topdown_camera_500' in self.robot.sensors:
            from internnav.evaluator.utils.common import norm_depth

            cur_obs = self.robot.sensors['topdown_camera_500'].get_data()
            obs['topdown_rgb'] = cur_obs['rgba'][..., :3]
            obs['topdown_depth'] = norm_depth(cur_obs['depth'])
        return obs
