import torch
from utils import set_seed
import os
from policy import ACTPolicy
import pickle
import time
from constants import FPS, PUPPET_GRIPPER_JOINT_OPEN
import numpy as np
from einops import rearrange


controller_policy_config = {
    'lr': 1e-5,
    'num_queries': 100, # chunk size
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    'vq': False,
    'vq_class': 0,
    'vq_dim': 0,
    'action_dim': 16,
    'no_encoder': False,
}
controller_config = {
    'ckpt_dir': '/media/aloha/DA51-1AE6/test_ckpt',
    'state_dim': 14,
    'policy_config': controller_policy_config,
    'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    'episode_len': 800,
    'temporal_agg': False
}


class SingleActionController():
    def __init__(self, config, ckpt_name):
        set_seed(1000)
        ckpt_dir = config['ckpt_dir']
        policy_config = config['policy_config']
        self.camera_names = config['camera_names']
        self.max_timesteps = config['episode_len']
        self.temporal_agg = config['temporal_agg']

        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = ACTPolicy(policy_config)
        self.policy.deserialize(torch.load(ckpt_path))
        self.policy.cuda()
        self.policy.eval()

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # load environment
        from aloha.robot_utils import move_grippers
        from aloha.real_env import make_real_env
        self.move_grippers = move_grippers
        self.robot = make_real_env(setup_robots=True, setup_base=True)

        # config temporal aggregation
        self.query_frequency = policy_config['num_queries']
        self.BASE_DELAY = 13
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = policy_config['num_queries'] - self.BASE_DELAY
        self.query_frequency -= self.BASE_DELAY

        # set time to run
        self.max_timesteps = int(self.max_timesteps * 1) # may increase for real-world tasks

    def reset(self):
        return self.robot.reset()

    def run(self):
        # home the arms
        ts = self.reset()
        # create storage for history and aggregation
        if self.temporal_agg:
            all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, 16]).cuda()

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(self.max_timesteps):
                time1 = time.time()
                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                qpos = self.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                if t % self.query_frequency == 0:
                    curr_image = self.get_image(ts, self.camera_names)

                # warm up
                if t == 0:
                    for _ in range(10):
                        self.policy(qpos, curr_image)
                    time1 = time.time()

                ### query policy
                if t % self.query_frequency == 0:
                    all_actions = self.policy(qpos, curr_image)
                    all_actions = torch.cat(
                        [all_actions[:, :-self.BASE_DELAY, :-2], all_actions[:, self.BASE_DELAY:, -2:]],
                        dim=2
                    )

                if self.temporal_agg:
                    all_time_actions[[t], t:t+self.num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % self.query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:]

                ### step the environment
                ts = self.robot.step(target_qpos, base_action)

                ### for visualization
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {self.max_timesteps / (time.time() - time0)}')

            self.move_grippers([self.robot.puppet_bot_left, self.robot.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open

    def get_image(self, ts, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image


if __name__ == "__main__":
    # have to run with this for now
    # python3 controller.py --ckpt_dir /media/aloha/DA51-1AE6/ --policy_class ACT --task_name test --seed 1000 --num_steps 8000
    sac = SingleActionController(controller_config, 'policy_best.ckpt')
    sac.run()