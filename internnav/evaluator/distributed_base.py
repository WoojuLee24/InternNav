import json
import os

import numpy as np
import torch

from internnav.configs.evaluator import EvalCfg
from internnav.env import Env
from internnav.evaluator import Evaluator
from internnav.utils.dist import (
    dist,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
)


class DistributedEvaluator(Evaluator):
    """
    Base class of distributed evaluators.

    Args:
        eval_cfg (EvalCfg): Evaluation configuration
        init_env (bool): Whether to initialize the environment
        init_agent (bool): Whether to initialize the agent
    """

    def __init__(self, eval_cfg: EvalCfg, init_env: bool = True, init_agent: bool = True):
        # distributed setting
        if not eval_cfg.eval_settings.get('use_agent_server', False):
            self.local_rank = init_distributed_mode(
                dist_url=eval_cfg.eval_settings.get('dist_url', "env://"),
                port=eval_cfg.eval_settings.get('port', 29529),
            )
        else:
            self.local_rank = 0
        np.random.seed(self.local_rank)

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.output_path = eval_cfg.eval_settings.get("output_path")

        # habitat env also need rank to split dataset
        eval_cfg.env.env_settings['rank'] = get_rank()
        eval_cfg.env.env_settings['local_rank'] = self.local_rank
        eval_cfg.env.env_settings['world_size'] = get_world_size()

        self.eval_config = eval_cfg

        if init_env:
            self.env = Env.init(eval_cfg.env, eval_cfg.task)

        # -------- initialize agent config (either remote server or local agent) --------
        if init_agent:
            if eval_cfg.eval_settings.get('use_agent_server', False):
                assert not is_dist_avail_and_initialized(), "agent server requires single evaluator process."
                # set agent port based on rank
                from internnav.utils import AgentClient

                print(f"[R{self.rank}] Connecting to agent server at port {eval_cfg.agent.server_port}")
                self.agent = AgentClient(eval_cfg.agent)
            else:
                from internnav.agent import Agent

                eval_cfg.agent.model_settings['local_rank'] = self.local_rank
                eval_cfg.agent.model_settings['task_name'] = eval_cfg.task.task_name
                self.agent = Agent.init(eval_cfg.agent)

    def eval(self):
        """
        Uniform distributed evaluation pipeline:

        1. Call subclass's eval_action() to get local per-episode tensors.
        2. Use dist all_gather (+ padding) to build global tensors for each metric.
        3. Call subclass's calc_metrics(global_metrics) to compute scalar metrics.
        4. Print + rank 0 writes result.json.
        """
        local_metrics = self.eval_action()  # dict[str, Tensor], each [N_local]

        if not local_metrics:
            raise RuntimeError("eval_action() returned empty metrics dict.")

        first_tensor = next(iter(local_metrics.values()))
        device = first_tensor.device
        local_len = first_tensor.shape[0]

        world_size = get_world_size()

        # -------- 1) Handle non-distributed / world_size == 1 --------
        if world_size == 1:
            global_metrics = {name: tensor.detach().cpu() for name, tensor in local_metrics.items()}
            total_len = int(local_len)
        else:
            # -------- 2-3) Gather metrics from all ranks via CPU (avoids GPU OOM on max_len) --------
            local_data = {name: tensor.detach().cpu().tolist() for name, tensor in local_metrics.items()}
            all_data = [None] * world_size
            dist.all_gather_object(all_data, local_data)

            total_len = sum(len(next(iter(d.values()))) for d in all_data if d)
            global_metrics = {}
            for name in local_data:
                combined = []
                for rank_data in all_data:
                    combined.extend(rank_data.get(name, []))
                global_metrics[name] = torch.tensor(combined)

        # -------- 4) Let subclass compute final metrics from global tensors --------
        result_all = self.calc_metrics(global_metrics)
        result_all.setdefault("length", total_len)

        # -------- 5) Logging --------
        print(result_all)
        if get_rank() == 0:
            os.makedirs(self.output_path, exist_ok=True)
            out_path = os.path.join(self.output_path, "result.json")
            with open(out_path, "a") as f:
                f.write(json.dumps(result_all) + "\n")

            if self.eval_config.eval_settings.get("use_wandb", False):
                try:
                    import wandb

                    if wandb.run is None:
                        wandb.init(
                            project=self.eval_config.eval_settings.get("wandb_project", "internnav"),
                            name=self.eval_config.eval_settings.get("wandb_run_name", None),
                            config=self.eval_config.eval_settings,
                        )
                    wandb.log({f"test/{k}": v for k, v in result_all.items()})
                except ImportError:
                    print("[Warning] wandb not installed. Skipping wandb logging.")

        return result_all

    # ================= ABSTRACT HOOKS =================

    def eval_action(self) -> dict:
        """
        Run evaluation on this rank and return per-episode metrics.

        Returns:
            dict[str, torch.Tensor]
                Example:
                {
                    "sucs": tensor([0., 1., ...], device=...),
                    "spls": tensor([...]),
                    "oss": tensor([...]),
                    "nes": tensor([...]),
                }
        """
        raise NotImplementedError

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        Compute final scalar metrics from global per-episode tensors.

        Args:
            global_metrics : dict[str, torch.Tensor]
                For each metric name, a 1-D CPU tensor with all episodes across all ranks.
                Example:
                    {
                        "sucs": tensor([...], dtype=torch.float32),
                        "spls": tensor([...]),
                        ...
                    }

        Returns:
            dict[str, float]
                Final scalar metrics to log.
        """
        raise NotImplementedError
