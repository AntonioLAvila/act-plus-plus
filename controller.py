import torch
from utils import set_seed
import os
from policy import ExplicitACTPolicy
from constants import controller_config
import pickle
import time
from constants import FPS, PUPPET_GRIPPER_JOINT_OPEN
import numpy as np
from einops import rearrange
from aloha.robot_utils import move_grippers
from aloha.real_env import make_real_env


class SingleActionController():
    '''
    Config should be set to the values you used during training.
    If you changed more than just
        * chunk_size
        * hidden_dim
        * dim_feedforward
    you'll need to also change the ACTArgs class in constants.
    '''
    def __init__(self, config, env, ckpt_name='policy_best.ckpt'):
        set_seed(1000)
        ckpt_dir = config['ckpt_dir']
        self.camera_names = config['camera_names']
        self.max_timesteps = config['episode_len']
        self.temporal_agg = config['temporal_agg']
        chunk_size = config['chunk_size'] # num_queries
        hidden_dim = config['hidden_dim']
        dim_ff = config['dim_ff']

        # load policy
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = ExplicitACTPolicy(chunk_size, self.camera_names, hidden_dim, dim_ff)
        self.policy.deserialize(torch.load(ckpt_path))
        self.policy.cuda()
        self.policy.eval()

        # load stats
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # reference to env
        self.robot = env

        # config temporal aggregation
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = chunk_size
        else:
            self.query_frequency = chunk_size
            self.num_queries = None

        # set time to run
        self.max_timesteps = int(self.max_timesteps * 1) # may increase for real-world tasks
        self.DT = 1 / FPS

    def reset(self):
        return self.robot.reset()

    def run(self):
        # home the arms
        ts = self.reset()
        # create storage for history and aggregation
        if self.temporal_agg:
            all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, 16]).cuda()

        with torch.inference_mode():
            start_time = time.time()
            culmulated_delay = 0
            for t in range(self.max_timesteps):
                loop_time = time.time()
                
                # get q obs
                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                qpos = self.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                # warm up
                if t == 0:
                    curr_image = self.get_image(ts, self.camera_names)
                    for _ in range(10):
                        self.policy(qpos, curr_image)
                    loop_time = time.time()

                # query policy
                if t % self.query_frequency == 0:
                    curr_image = self.get_image(ts, self.camera_names)
                    all_actions = self.policy(qpos, curr_image)
                
                # assign action
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

                # post-process action
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:]
                
                # step the environment
                ts = self.robot.step(target_qpos, base_action)

                # keep pace
                duration = time.time() - loop_time
                sleep_time = max(0, self.DT - duration)
                time.sleep(sleep_time)

                # logging
                if duration >= self.DT:
                    culmulated_delay += (duration - self.DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {self.DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {self.max_timesteps / (time.time() - start_time)}')

    def get_image(self, ts, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image


    def open_grippers(self):
        move_grippers([self.robot.follower_bot_left, self.robot.follower_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, 0.5)


def dead_rekckoning_turn(robot):
    '''
    Turn right for 2 seconds at pi/6 rad/s for a 60 deg turn.
    We have no gyro :,) wtf
    '''
    arm_action, base_action = robot.get_qpos(), (0, -np.pi/6) # linear, angular
    robot.step(arm_action, base_action)
    time.sleep(2)
    robot.step(arm_action, (0, 0))
    

if __name__ == "__main__":
    robot = make_real_env(setup_robots=True, setup_base=True)
    sac = SingleActionController(controller_config, robot)
    sac.run()
    sac.open_grippers()