import habitat_sim
if not hasattr(habitat_sim.Agent, 'sensors'):
    habitat_sim.Agent.sensors = property(lambda x: x._sensors)

from internnav.habitat_extensions.vln.habitat_vln_evaluator import HabitatVLNEvaluator
