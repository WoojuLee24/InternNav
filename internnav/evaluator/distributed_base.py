import json
import os
import tempfile
import time

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
            # -------- 2) File-based metric collection (avoids NCCL OOM/timeout) --------
            # Each rank writes its metrics to a file; rank 0 polls and collects.
            # This replaces dist.all_gather_object which requires all ranks to call
            # simultaneously — ranks with fewer episodes finish early and time out.
            collect_dir = self.output_path or tempfile.mkdtemp()
            os.makedirs(collect_dir, exist_ok=True)

            rank = get_rank()
            local_data = {name: tensor.detach().cpu().tolist() for name, tensor in local_metrics.items()}
            rank_file = os.path.join(collect_dir, f"metrics_rank{rank}.json")
            tmp_file = rank_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(local_data, f)
            os.replace(tmp_file, rank_file)  # atomic write

            if rank != 0:
                return {}

            # Rank 0 waits for all rank files
            all_data = []
            for r in range(world_size):
                r_file = os.path.join(collect_dir, f"metrics_rank{r}.json")
                while not os.path.exists(r_file):
                    time.sleep(2)
                with open(r_file, "r") as f:
                    all_data.append(json.load(f))
            for r in range(world_size):
                try:
                    os.remove(os.path.join(collect_dir, f"metrics_rank{r}.json"))
                except OSError:
                    pass

            total_len = sum(len(next(iter(d.values()))) for d in all_data)
            global_metrics = {}
            for name in local_data:
                combined = []
                for rank_data in all_data:
                    combined.extend(rank_data.get(name, []))
                global_metrics[name] = torch.tensor(combined)

        # -------- 4) Let subclass compute final metrics from global tensors (rank 0 only) --------
        result_all = self.calc_metrics(global_metrics)
        result_all.setdefault("length", total_len)

        # -------- 5) Logging --------
        print(result_all)
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            out_path = os.path.join(self.output_path, "result.json")
            with open(out_path, "a") as f:
                f.write(json.dumps(result_all) + "\n")

        if self.eval_config.eval_settings.get("use_wandb", False):
            try:
                import wandb

                if wandb.run is None:
                    wandb.init(
                        project=self.eval_config.eval_settings.get("wandb_project", "huggingface"),
                        name=self.eval_config.eval_settings.get("wandb_run_name", None),
                        config=self.eval_config.eval_settings,
                    )
                log_dict = {f"test/{k}": v for k, v in result_all.items()}
                best_checkpoint = self.eval_config.eval_settings.get("best_checkpoint", None)
                if best_checkpoint is not None:
                    import re
                    m = re.search(r"(\d+)$", best_checkpoint)
                    if m:
                        log_dict["test/best_checkpoint_step"] = int(m.group(1))
                wandb.log(log_dict)
                wandb.finish()
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
