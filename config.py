conf = {
    "WORK_PATH": "/home/nsec0/zyx/GaitSet-master/work",
    "CUDA_VISIBLE_DEVICES": "0",  # "0,1,2,3",
    "data": {
        # 'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait4",
        'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait_noise/0.0",
        # 'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_VR5Gait",
        # 'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait",
        'resolution': '64',
        'dataset': 'TestSysGaitN', # 'TestSysGait', # 'SysGaitN',  # 'SysGait',  # 'CASIA-B', # VRGait # VR2Gait # 'VR5Gait'
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 0,  # 0, # 18,  # 73,  # should be larger than training batch_size=8
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',  # hard
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 80000,  #80000,  # 150000 for VR5Gait
        'margin': 0.2,
        'num_workers': 8,  # 3,
        'frame_num': 30,  # [NOTE] default:30, change here for factor 
        'model_name': 'SysGait4',  # 'SysGait4'
    },
}
