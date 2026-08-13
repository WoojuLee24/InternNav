import os
from enum import Enum
from pathlib import Path
from time import time
from typing import Dict, List

import numpy as np

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.evaluator.utils.common import set_seed_model
from internnav.evaluator.utils.config import get_lmdb_path
from internnav.evaluator.utils.data_collector import DataCollector
from internnav.evaluator.utils.result_logger import ResultLogger
from internnav.evaluator.utils.visualize_util import VisualizeUtil
from internnav.utils import common_log_util, progress_log_multi_util
from internnav.utils.common_log_util import common_logger as log


class runner_status_code(Enum):
    NORMAL = 0
    WARM_UP = 1
    NOT_RESET = 3
    TERMINATED = 2
    STOP = 4


@Evaluator.register('vln_distributed')
class VLNDistributedEvaluator(DistributedEvaluator):
    def __init__(self, config: EvalCfg):
        start_time = time()

        self.task_name = config.task.task_name
        self.result_logger = ResultLogger(config.dataset)
        self.dataset_name = Path(config.dataset.dataset_settings['base_data_dir']).name
        config.env.env_settings['dataset'] = config.dataset

        # vec env settings
        self.env_num = config.task.task_settings['env_num']
        self.proc_num = (
            config.env.env_settings['distribution_config']['proc_num']
            if 'distribution_config' in config.env.env_settings
            else 1
        )

        # update config
        config.task.task_settings['env_num'] = self.env_num
        if 'distribution_config' in config.env.env_settings:
            config.env.env_settings['distribution_config']['proc_num'] = self.proc_num

        config.agent.model_settings.update({'env_num': self.env_num, 'proc_num': self.proc_num})
        self.robot_name = config.task.robot_name

        super().__init__(config)
        set_seed_model(0)

        common_log_util.init(self.task_name)
        self.total_path_num = len(self.env.episodes)
        progress_log_multi_util.init(self.task_name, self.total_path_num)
        progress_log_multi_util.progress_logger_multi.info(
            f'start eval dataset: {self.task_name}, total_path: {self.total_path_num}'  # noqa: E501
        )
        self.data_collector = DataCollector(get_lmdb_path(self.task_name), rank=self.rank, world_size=self.world_size)
        self.robot_flash = config.task.robot_flash
        self.save_to_json = config.eval_settings['save_to_json']
        self.vis_output = config.eval_settings['vis_output']
        self.show_rgb = config.eval_settings.get('show_rgb', False)
        self.visualize_util = VisualizeUtil(self.task_name, fps=6)

        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] Env Init time: {duration}s')

    @property
    def ignore_obs_attr(self):
        return [
            'finish_action',
            'current_pose',
            'render',
            'fail_reason',
            'metrics',
        ]

    def remove_obs_attr(self, obs):
        return [{k: v for k, v in ob.items() if k not in self.ignore_obs_attr} for ob in obs]

    def warm_up(self):
        while True:
            obs, _, _, _, _ = self.env.step(
                action=[{self.robot_name: {'stand_still': []}} for _ in range(self.env_num * self.proc_num)]
            )
            if obs[0][self.robot_name]['finish_action']:
                break
        return obs

    def now_path_key(self, info):
        return info.data['path_key']

    def _obs_remove_robot_name(self, obs):
        obs = [
            *map(
                lambda ob: ob[self.robot_name] if ob is not None else self.fake_obs,
                obs,
            )
        ]
        return obs

    def _transform_action_batch(self, actions: List[Dict], flash=False):
        transformed_actions = []
        for action in actions:
            if 'ideal_flag' in action.keys():
                ideal_flag = action['ideal_flag']
                if flash:
                    assert ideal_flag is True
            else:
                ideal_flag = False
            if not ideal_flag:
                transformed_actions.append({'h1': {'vln_dp_move_by_speed': action['action'][0]}})
                continue
            a = action['action']
            if a == 0 or a == [0] or a == [[0]]:
                transformed_actions.append({'h1': {'stop': []}})
            elif a == -1 or a == [-1] or a == [[-1]]:
                transformed_actions.append({'h1': {'stand_still': []}})
            else:
                move = f"move_by_{'discrete' if not flash else 'flash'}"
                transformed_actions.append({'h1': {move: a}})  # discrete e.g. [3]
        return transformed_actions

    def get_action(self, obs, action):
        start_time = time()
        # process obs
        obs = np.array(obs)
        fake_obs_index = np.logical_or(
            self.runner_status == runner_status_code.WARM_UP,
            self.runner_status == runner_status_code.TERMINATED,
        )
        obs[fake_obs_index] = self.fake_obs
        obs = self.remove_obs_attr(obs)
        if not np.logical_and.reduce(self.runner_status == runner_status_code.WARM_UP):
            action = self.agent.step(obs)
            log.info(f'now action: {len(action)}, {action}, fake_obs_index: {fake_obs_index}')
            action = self._transform_action_batch(action, self.robot_flash)
        # change warm_up
        action = np.array(action)
        action[self.runner_status == runner_status_code.WARM_UP] = {'h1': {'stand_still': []}}
        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] agent step time: {duration}s')
        return obs, action

    def _need_reset(self, terminated_ls):
        return np.logical_or.reduce(
            np.logical_and(
                terminated_ls,
                (self.runner_status != runner_status_code.TERMINATED),
            )
        )

    def env_step(self, action):
        start_time = time()

        while True:
            # stop action maybe also need 50 steps
            self.runner_status[
                np.logical_and(self.runner_status == runner_status_code.NORMAL, action == {'h1': {'stop': []}})
            ] = runner_status_code.STOP
            obs, reward, terminated, truncated, info = self.env.step(action=action.tolist())
            obs = self._obs_remove_robot_name(obs)
            finish_status = np.logical_or(
                np.array([ob['finish_action'] for ob in obs]),
                np.array(terminated),
            )  # strong condition

            if (
                np.logical_and.reduce(np.array(finish_status)[self.runner_status == runner_status_code.NORMAL])
                and runner_status_code.NORMAL in self.runner_status
            ) or np.logical_and.reduce(np.array(finish_status)):
                self.runner_status[self.runner_status == runner_status_code.STOP] = runner_status_code.NORMAL
                break
        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] Env Step time: {duration}s')
        return obs, terminated

    def terminate_ops(self, obs_ls, reset_infos, terminated_ls):
        """
        1. reset agent if finished warm up
        2. reset envs that are terminated
        3. start new trace log and visualize log
        4. return whether all envs are terminated
        5. return updated reset_infos
        """
        start_time = time()

        finish_warmup_ls = (self.runner_status == runner_status_code.WARM_UP) & [ob['finish_action'] for ob in obs_ls]
        if np.logical_or.reduce(finish_warmup_ls):
            self.agent.reset(np.where(finish_warmup_ls)[0].tolist())
            self.runner_status[finish_warmup_ls] = runner_status_code.NORMAL
            log.info(f'env{np.where(finish_warmup_ls)[0].tolist()}: states switch to NORMAL.')
        # if no need reset, return False
        if not self._need_reset(terminated_ls):
            return False, reset_infos
        import json

        for env_id, terminated in enumerate(terminated_ls):
            if terminated and self.runner_status[env_id] != runner_status_code.TERMINATED:
                obs = obs_ls[env_id]
                reset_info = reset_infos[env_id]
                log.info(f"{self.now_path_key(reset_info)}: {json.dumps(obs['metrics'], indent=4)}")
                self.data_collector.save_eval_result(
                    key=self.now_path_key(reset_info),
                    result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                    info=obs['metrics'][list(obs['metrics'].keys())[0]][0],
                )  # save data to dataset
                # log data
                progress_log_multi_util.trace_end(
                    trajectory_id=self.now_path_key(reset_info),
                    step_count=obs['metrics'][list(obs['metrics'].keys())[0]][0]['steps'],
                    result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                )
                # visualize
                if self.vis_output:
                    self.visualize_util.trace_end(
                        trajectory_id=self.now_path_key(reset_info),
                        result=obs['metrics'][list(obs['metrics'].keys())[0]][0]['fail_reason'],
                    )
                # json format result
                result = self.result_logger.finalize_all_results(self.rank, self.world_size)
                if self._wandb_active and result:
                    try:
                        import wandb
                        collision_tag = self.eval_config.task.flash_collision or "none"
                        # running metrics under the SAME test_<split>_<tag>/ keys as the
                        # final log below -- one wandb section for the whole eval. No step=
                        # arg: a resumed run's step is already past `count`, which silently
                        # dropped these logs before.
                        for split, metrics in result.items():
                            wandb.log({f"test_{split}_{collision_tag}/{k}": v for k, v in metrics.items()})
                    except Exception as e:
                        print(f"[Warning] wandb step log failed: {e}")
                self.runner_status[env_id] = runner_status_code.NOT_RESET
                log.info(f'env{env_id}: states switch to NOT_RESET.')
        # need this status to reset
        reset_env_ids = np.where(self.runner_status == runner_status_code.NOT_RESET)[0].tolist()
        if len(reset_env_ids) > 0:
            log.info(f'env{reset_env_ids}: start new episode!')
            obs, new_reset_infos = self.env.reset(reset_env_ids)
            self.runner_status[reset_env_ids] = runner_status_code.WARM_UP
            log.info(f'env{reset_env_ids}: states switch to WARM UP.')

            # modify original reset_info
            reset_infos = np.array(reset_infos)
            # If there is only one reset and no new_deset_infos, return an empty array
            reset_infos[reset_env_ids] = new_reset_infos if len(new_reset_infos) > 0 else None
            self.runner_status[
                np.vectorize(lambda x: x)(reset_infos) == None  # noqa: E711
            ] = runner_status_code.TERMINATED
            log.info(f'env{np.vectorize(lambda x: x)(reset_infos) == None}: states switch to TERMINATED.')
            reset_infos = reset_infos.tolist()

        if np.logical_and.reduce(self.runner_status == runner_status_code.TERMINATED):
            return True, reset_infos
        for reset_info in new_reset_infos:
            if reset_info is None:
                continue
            # start new trace log
            progress_log_multi_util.trace_start(
                trajectory_id=self.now_path_key(reset_info),
            )
            # start new visualize log
            if self.vis_output:
                self.visualize_util.trace_start(
                    trajectory_id=self.now_path_key(reset_info), reference_path=reset_info.data['reference_path']
                )

        end_time = time()
        duration = round(end_time - start_time, 2)
        log.info(f'[TIME] Env Reset time: {duration}s')
        return False, reset_infos

    def _wandb_init(self):
        if not self.eval_config.eval_settings.get("use_wandb", True):
            return False
        try:
            import wandb
            if wandb.run is None:
                wandb.init(
                    entity=self.eval_config.eval_settings.get("wandb_entity"),
                    project=self.eval_config.eval_settings.get("wandb_project", "huggingface"),
                    name=self.eval_config.eval_settings.get("wandb_run_name"),
                )
            # Log Count=0 at startup so the test_<split>_<tag>/ panel exists in the wandb
            # UI right away -- otherwise the first evidence that logging works is the
            # first finished episode, minutes later. Same keys as the per-episode and
            # final logs, so this adds no extra wandb section.
            collision_tag = getattr(self.eval_config.task, "flash_collision", None) or "none"
            for split in self.result_logger.split_map:
                wandb.log({f"test_{split}_{collision_tag}/Count": 0})
            print(f"[wandb] eval logging -> {wandb.run.url} "
                  f"(project={wandb.run.project}, name={wandb.run.name}, step={wandb.run.step})",
                  flush=True)
            return True
        except Exception as e:
            print(f"[Warning] wandb init failed: {e}")
            return False

    def eval(self):
        print('--- VlnMultiEvaluator start ---')
        self._wandb_active = self._wandb_init()
        obs, reset_info = self.env.reset()
        for info in reset_info:
            if info is None:
                continue
            progress_log_multi_util.trace_start(
                trajectory_id=self.now_path_key(info),
            )
            if self.vis_output:
                self.visualize_util.trace_start(
                    trajectory_id=self.now_path_key(info), reference_path=info.data['reference_path']
                )
        log.info('start new episode!')

        obs = self.warm_up()
        self.fake_obs = obs[0][self.robot_name]
        action = [{self.robot_name: {'stand_still': []}} for _ in range(self.env_num * self.proc_num)]
        obs = self._obs_remove_robot_name(obs)
        self.runner_status = np.full(
            (self.env_num * self.proc_num),
            runner_status_code.NORMAL,
            runner_status_code,
        )
        self.runner_status[[info is None for info in reset_info]] = runner_status_code.TERMINATED

        while self.env.is_running():
            # get action from agent
            obs, action = self.get_action(obs, action)
            # step env
            obs, terminated = self.env_step(action)
            # terminate ops
            env_terminate, reset_info = self.terminate_ops(obs, reset_info, terminated)

            if env_terminate:
                break

            # debugpy: early exit after 2 episodes for quick wandb connectivity check
            if os.environ.get("DEBUGPY") and self.result_logger.last_result:
                count = max((m.get("Count", 0) for m in self.result_logger.last_result.values()), default=0)
                if count >= 2:
                    print(f"[debugpy] Early exit after {count} episodes.", flush=True)
                    break

            # show live RGB from env 0
            if self.show_rgb:
                import cv2
                for i, ob in enumerate(obs):
                    if ob is None or 'rgb' not in ob:
                        continue
                    frame = cv2.cvtColor(ob['rgb'].copy(), cv2.COLOR_RGB2BGR)
                    h, w = frame.shape[:2]

                    # overlay instruction text
                    if 'instruction' in ob:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale, thickness = 0.45, 1
                        words = ob['instruction'].split()
                        lines, cur = [], ''
                        for word in words:
                            trial = (cur + ' ' + word).strip()
                            (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
                            if tw > w - 10 and cur:
                                lines.append(cur)
                                cur = word
                            else:
                                cur = trial
                        if cur:
                            lines.append(cur)
                        for j, line in enumerate(lines):
                            y = 14 + j * 16
                            cv2.rectangle(frame, (0, y - 12), (w, y + 4), (0, 0, 0), -1)
                            cv2.putText(frame, line, (5, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    # overlay trajectory minimap (reference path + current robot position)
                    info = reset_info[i] if i < len(reset_info) else None
                    if info is not None and 'reference_path' in info.data and 'globalgps' in ob:
                        ref_pts = np.array(info.data['reference_path'])[:, :2]
                        cur_xy = np.array(ob['globalgps'])[:2]
                        all_pts = np.vstack([ref_pts, cur_xy.reshape(1, 2)])
                        mn, mx = all_pts.min(0), all_pts.max(0)
                        span = np.maximum(mx - mn, 1e-6)
                        map_sz, pad = 150, 10
                        scale = (map_sz - 2 * pad) / span.max()

                        def to_px(pt):
                            px = int((pt[0] - mn[0]) * scale) + pad
                            py = map_sz - pad - int((pt[1] - mn[1]) * scale)
                            return (int(np.clip(px, 0, map_sz - 1)), int(np.clip(py, 0, map_sz - 1)))

                        mmap = np.full((map_sz, map_sz, 3), 40, dtype=np.uint8)
                        for k in range(len(ref_pts) - 1):
                            cv2.line(mmap, to_px(ref_pts[k]), to_px(ref_pts[k + 1]), (0, 200, 0), 1)
                        for pt in ref_pts:
                            cv2.circle(mmap, to_px(pt), 2, (0, 255, 0), -1)
                        cv2.circle(mmap, to_px(cur_xy), 5, (0, 80, 255), -1)
                        x0, y0 = w - map_sz - 5, h - map_sz - 5
                        frame[y0:y0 + map_sz, x0:x0 + map_sz] = mmap

                    cv2.imshow('Robot RGB', frame)
                    cv2.waitKey(1)
                    break

            # save step obs
            if self.vis_output:
                for ob, info, act in zip(obs, reset_info, action):
                    if info is None or 'rgb' not in ob or ob['fail_reason']:
                        continue
                    self.visualize_util.save_observation(
                        trajectory_id=self.now_path_key(info), obs=ob, action=act[self.robot_name]
                    )

        # NOTE: log the final metrics BEFORE env.close(). On h1 (Isaac Sim) closing the
        # env tears down the Simulation App and the process dies inside it, so nothing
        # after env.close() ever runs -- that is why these test_* metrics used to be
        # missing from wandb and had to be backfilled from result_h1.json.
        if self._wandb_active and self.result_logger.last_result:
            try:
                import wandb
                collision_tag = getattr(self.eval_config.task, "flash_collision", None) or "none"
                for split, metrics in self.result_logger.last_result.items():
                    wandb.log({f"test_{split}_{collision_tag}/{k}": v for k, v in metrics.items()})
                wandb.finish()
            except Exception as e:
                print(f"[Warning] wandb logging failed: {e}")

        self.env.close()
        progress_log_multi_util.report()

        print('--- VlnMultiEvaluator end ---')
