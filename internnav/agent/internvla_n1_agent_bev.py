"""InternVLA-N1 agent with BEV visual input (Isaac Sim / H1 path).

Subclass of ``InternVLAN1Agent`` — internvla_n1_agent.py is NOT modified.
Selected via ``model_name='internvla_n1_bev'``; all BEV behaviour lives in
``InternVLAN1NetBEV`` (the policy), so ``step()`` is inherited unchanged.

The parent ``__init__`` instantiates the policy through the module-level
``get_policy``/``get_config`` factory. To avoid loading the multi-GB model
twice, we temporarily redirect ``get_policy`` to return ``InternVLAN1NetBEV``
while the parent constructor runs (same monkey-patch precedent as
internnav/trainer/internvla_n1_bev_trainer.py), then restore it.

NOTE: this module must be imported for the registry entry to exist — the BEV
config files do this with ``import internnav.agent.internvla_n1_agent_bev``.
"""

import internnav.agent.internvla_n1_agent as _base_module
from internnav.agent.base import Agent
from internnav.agent.internvla_n1_agent import InternVLAN1Agent
from internnav.configs.agent import AgentCfg
from internnav.model.basemodel.internvla_n1.internvla_n1_policy_bev import InternVLAN1NetBEV


@Agent.register('internvla_n1_bev')
class InternVLAN1AgentBEV(InternVLAN1Agent):
    """InternVLAN1Agent whose policy is InternVLAN1NetBEV (provider-aware)."""

    def __init__(self, config: AgentCfg):
        original_get_policy = _base_module.get_policy
        _base_module.get_policy = lambda policy_name: InternVLAN1NetBEV
        try:
            # get_config stays untouched — InternVLAN1NetBEV uses the parent's
            # InternVLAN1ModelConfig (keep policy_name='InternVLAN1_Policy').
            super().__init__(config)
        finally:
            _base_module.get_policy = original_get_policy
