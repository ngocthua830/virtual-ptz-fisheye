"""Scenario config for the single-fisheye PTZ environment."""

PTZ_ACTIONS = {
    'STAY':      0,
    'PAN_LEFT':  1,
    'PAN_RIGHT': 2,
    'ZOOM_IN':   3,
    'ZOOM_OUT':  4,
    'TILT_UP':   5,   # toward horizon
    'TILT_DOWN': 6,   # toward nadir
}
NUM_ACTIONS = 7

FISHEYE_CONFIG = {
    'data_path': './data/1767931881294.mp4',
    'fish_fov_deg': 180.0,
    'frame_skip': 5,
}

PTZ_CONFIG = {
    'pan_range':  (-180.0, 180.0),     # wraps around
    'tilt_range': (0.0, 80.0),         # 0 = straight down (nadir)
    'zoom_range': (1.0, 8.0),
    'pan_speed':  25.0,
    'tilt_speed': 15.0,
    'zoom_speed': 0.5,
    'base_fov':   90.0,                # perspective FOV at zoom=1
    'ptz_out_w':  960,
    'ptz_out_h':  540,
}

STATE_CONFIG = {
    'state_dim': 16,
}

TRAINING_CONFIG = {
    'num_episodes': 500,
    'max_steps': 128,
    'batch_size': 256,
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.02,
    'epsilon_decay': 5000,
    'actor_lr': 1e-4,
    'param_net_lr': 1e-5,
    'replay_memory_size': 100000,
    'target_update_freq': 100,
}

REWARD_CONFIG = {
    'tracking_reward': 1.0,
    'detection_bonus': 0.5,
    'center_bonus': 0.3,
    'action_penalty': -0.05,
    'miss_penalty': -0.5,
}

YOLO_CONFIG = {
    'model_name': 'yolov8n.pt',
    'conf_thresh': 0.4,
    'iou_thresh': 0.5,
    'device': 'cuda',
    'target_classes': ['person'],
}
