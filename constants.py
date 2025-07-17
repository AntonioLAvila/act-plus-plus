import pathlib
import getpass

# assumes data is stored at ~/aloha_data
user = getpass.getuser()

DATA_DIR = '/home/' + user + '/aloha_data'
if user == 'aavila':
    DATA_DIR = '/mnt/home_mnt/home_mnt/aavila/aloha_data'

### Task parameters
if getpass.getuser() == 'aloha':
    from aloha.constants import TASK_CONFIGS
else:
    TASK_CONFIGS = {
        'test':{
            'dataset_dir': DATA_DIR + '/test',
            'num_episodes': 1,
            'episode_len': 800,
            'camera_names':['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        },
        'move_wafer': {
            'dataset_dir': DATA_DIR + '/move_wafer',
            'episode_len': 1600,
            'camera_names':['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        },
        'pick_up_wafer': {
            'dataset_dir': DATA_DIR + '/pick_up_wafer',
            'episode_len': 550,
            'camera_names':['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        }
    }

controller_config = {
    'ckpt_dir': '/media/' + user + '/DA51-1AE6/test_ckpt',              # from you
    'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],  # from TASK_CONFIG (always going to be this though)
    'episode_len': 800,                                                 # from TASK_CONFIG
    'temporal_agg': False,                                              # up to you, only affects runtime
    'chunk_size': 100,                                                  # set at training time
    'hidden_dim': 512,                                                  # set at training time
    'dim_ff': 3200                                                      # set at training time
}

# If you tune the model's other parameters like dilation for example
# when creating the controller you'll need to also change these
class ACTArgs():
    def __init__(self):
        # theres also a hard coded state dim in build_ACT_model
        # main tunable parameters
        self.num_queries = None
        self.camera_names = None
        self.hidden_dim = None
        self.dim_feedforward = None
        # parameters
        self.nheads = 8
        self.enc_layers = 4
        self.dec_layers = 7
        # should probably stay this value
        self.action_dim = 16
        self.lr_backbone = 1e-5
        self.backbone = 'resnet18'
        self.dilation = False
        self.masks = False
        self.position_embedding = 'sine'
        self.dropout = 0.1
        self.pre_norm = False
        self.no_encoder = False
        # vq unsupported
        self.vq = False
        self.vq_class = 0
        self.vq_dim = 0


### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
