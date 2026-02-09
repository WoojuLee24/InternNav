from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient

# initialize agent
agent=AgentCfg(
      server_host='localhost',
      server_port=8087,
      model_name='internvla_n1',
      ckpt_path='',
      model_settings={
            'policy_name': "InternVLAN1_Policy",
            'state_encoder': None,
            'env_num': 1,
            'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            'vis_debug': True,
            'vis_debug_path': 'vis_debug',
      },
      vis_debug=True
)
agent = AgentClient(agent)

# Load a capture from saved D455 camera:
from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta
rs_meta_path = '/home/gdr/gd_vln/workspace/src/InternNav/scripts/iros_challenge/onsite_competition/captures/rs_meta.json'

fake_obs_640 = load_obs_from_meta(rs_meta_path)
fake_obs_640['instruction'] = 'go to the red car'
print(fake_obs_640['rgb'].shape, fake_obs_640['depth'].shape)


# Test model inference
action = agent.step([fake_obs_640])[0]['action'][0]
print(f"Action taken: {action}")



