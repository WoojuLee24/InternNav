"""InternVLA-N1 agent with unified (view/type/mode/combine) visual input (Isaac Sim / H1 path).

Subclass of ``InternVLAN1Agent`` — internvla_n1_agent.py is NOT modified. All
unified-provider behaviour lives in ``InternVLAN1NetUnified`` (the policy); ``step()``
adds exactly one thing on top of the inherited behaviour: snapshotting the
ground-truth topdown debug fields onto the policy, since ``InternVLAN1Agent.step(self, obs)``
is the only place in the whole call chain with access to the full obs dict
(``s1_step_latent``/``s2_step`` only ever receive ``rgb``/``depth``).

The parent ``__init__`` instantiates the policy through the module-level
``get_policy``/``get_config`` factory. To avoid loading the multi-GB model twice, we
temporarily redirect ``get_policy`` to return ``InternVLAN1NetUnified`` while the parent
constructor runs (same monkey-patch precedent as ``internvla_n1_agent_bev.py``), then
restore it.

NOTE: this module must be imported for the registry entry to exist — the image_base
Isaac config path does this via ``default_config.py``'s ``build_h1_eval_cfg``.
"""

import internnav.agent.internvla_n1_agent as _base_module
from internnav.agent.base import Agent
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
from internnav.configs.agent import AgentCfg
from internnav.model.basemodel.internvla_n1.internvla_n1_policy_unified import InternVLAN1NetUnified


@Agent.register('internvla_n1_unified')
class InternVLAN1AgentUnified(InternVLAN1Agent):
    """InternVLAN1Agent whose policy is InternVLAN1NetUnified (provider-aware)."""

    def __init__(self, config: AgentCfg):
        original_get_policy = _base_module.get_policy
        _base_module.get_policy = lambda policy_name: InternVLAN1NetUnified
        try:
            # get_config stays untouched — InternVLAN1NetUnified uses the parent's
            # InternVLAN1ModelConfig (keep policy_name='InternVLAN1_Policy').
            super().__init__(config)
        finally:
            _base_module.get_policy = original_get_policy

    def step(self, obs):
        # obs is a List[dict] (see InternVLAN1Agent.step's own `obs = obs[0]`); index the
        # same way here before the parent method does its own obs[0] lookup.
        self.policy.set_debug_topdown(obs[0].get('topdown_rgb'), obs[0].get('globalrotation'))
        return super().step(obs)
