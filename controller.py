import torch
from utils import set_seed
import os
from policy import ACTPolicy
import pickle
import time
from constants import FPS, PUPPET_GRIPPER_JOINT_OPEN
import numpy as np
from einops import rearrange
from constants import defualt_eval_config
import sys
sys.path.append('/home/aloha/interbotix_ws/src')


def controller(config, ckpt_name):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    from aloha.aloha.robot_utils import move_grippers # requires aloha
    from aloha.aloha.real_env import make_real_env # requires aloha
    env = make_real_env(init_node=True, setup_robots=True, setup_base=True)

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    BASE_DELAY = 13
    query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    ts = env.reset()

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()

    # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    qpos_history_raw = np.zeros((max_timesteps, state_dim))
    qpos_list = []
    target_qpos_list = []
    with torch.inference_mode():
        time0 = time.time()
        DT = 1 / FPS
        culmulated_delay = 0
        for t in range(max_timesteps):
            time1 = time.time()
            ### process previous timestep to get qpos and image_list
            time2 = time.time()
            obs = ts.observation
            qpos_numpy = np.array(obs['qpos'])
            qpos_history_raw[t] = qpos_numpy
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            # qpos_history[:, t] = qpos
            if t % query_frequency == 0:
                curr_image = get_image(ts, camera_names)
            # print('get image: ', time.time() - time2)

            if t == 0:
                # warm up
                for _ in range(10):
                    policy(qpos, curr_image)
                print('network warm up done')
                time1 = time.time()

            ### query policy
            time3 = time.time()
            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)
                all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
            if temporal_agg:
                all_time_actions[[t], t:t+num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]
                # if t % query_frequency == query_frequency - 1:
                #     # zero out base actions to avoid overshooting
                #     raw_action[0, -2:] = 0
            # print('query policy: ', time.time() - time3)

            ### post-process actions
            time4 = time.time()
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action[:-2]
            base_action = action[-2:]

            ### step the environment
            time5 = time.time()
            ts = env.step(target_qpos, base_action)
            # print('step env: ', time.time() - time5)

            ### for visualization
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            duration = time.time() - time1
            sleep_time = max(0, DT - duration)
            # print(sleep_time)
            time.sleep(sleep_time)
            # time.sleep(max(0, DT - duration - culmulated_delay))
            if duration >= DT:
                culmulated_delay += (duration - DT)
                print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
            # else:
            #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

        print(f'Avg fps {max_timesteps / (time.time() - time0)}')

        move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
        # save qpos_history_raw
        log_id = get_auto_index(ckpt_dir)
        np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    
    return curr_image


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == "__main__":
    controller(defualt_eval_config, 'test')