from datetime import datetime
import numpy as np
import argparse
import os
from model.initialization import initialization
from model.utils import *
# from config import conf
import torch
import torch.nn.functional as F



def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    # print("--Debug-- dist shape {}".format(dist.shape))

    return dist

def get_npys(view_list, frame_list):
    
    conf = {
        "WORK_PATH": "/home/nsec0/zyx/GaitSet-master/work",
        "CUDA_VISIBLE_DEVICES": "3",  # "0,1,2,3",
        "data": {
            'dataset_path': "/home/nsec0/zyx/GaitSet-master/pretreat_VR5Gait",
            'resolution': '64',
            'dataset': 'VR5Gait',  # 'SysGait',  # 'CASIA-B', # VRGait # VR2Gait # 'VR5Gait'
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
            'total_iter': 150000,  #80000,  # 150000 for VR5Gait
            'margin': 0.2,
            'num_workers': 8,  # 3,
            'frame_num': 30,  # [NOTE] default:30, change here for factor 
            'model_name': 'SysGait4',  # 'SysGait4'
        },
    }

    for view_config in view_list:
        for frame_config in frame_list:
            
            if view_config == 'FBRL':
                conf["data"]['dataset_path'] = "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_VR5Gait"
            else:
                conf["data"]['dataset_path'] = "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_VR5Gait_{}".format(view_config)

            conf["model"]['frame_num'] = frame_config

            suffix = "_{}_Frame{}".format(view_config, frame_config)

            time1 = datetime.now()
            m = initialization(conf, test=False)[0]
            m.load(80000)

            time2 = datetime.now()

            test_data = m.transform('test', 1)
            time3 = datetime.now()

            feature, view, seq_type, label = test_data

            np.save("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait{}.npy".format(suffix), feature)
            np.save("/home/nsec0/zyx/GaitSet-master/infer_all_view_VR5Gait{}.npy".format(suffix), view)
            np.save("/home/nsec0/zyx/GaitSet-master/infer_all_seqtype_VR5Gait{}.npy".format(suffix), seq_type)
            np.save("/home/nsec0/zyx/GaitSet-master/infer_all_label_VR5Gait{}.npy".format(suffix), label)

            print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))
            print("--!!--features saved")
            print("model init and load: {}, feature extrac(for 10vol*10mh*10seq*4view):{}".format(time2-time1, time3-time2))

# get_npys(view_list=['FBRL', ], frame_list=range(1, 129))
# get_npys(view_list=['F', 'FB', 'FBR', 'FR', 'FRL', 'R', 'RL', 'FBRL'], frame_list=[30, ])


def get_npys_231224():
    
    conf = {
        "WORK_PATH": "/home/nsec0/zyx/GaitSet-master/work",
        "CUDA_VISIBLE_DEVICES": "3",  # "0,1,2,3",
        "data": {
            # 'dataset_path': "/home/nsec0/zyx/GaitSet-master/pretreat_VR5Gait",
            'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait",
            'resolution': '64',
            'dataset': 'TestSysGait',  # 'SysGait',  # 'CASIA-B', # VRGait # VR2Gait # 'VR5Gait'
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
            'total_iter': 150000,  #80000,  # 150000 for VR5Gait
            'margin': 0.2,
            'num_workers': 8,  # 3,
            'frame_num': 30,  # [NOTE] default:30, change here for factor 
            'model_name': 'SysGait4',  # 'SysGait4'
        },
    }

    time1 = datetime.now()
    m = initialization(conf, test=False)[0]
    m.load(80000)

    time2 = datetime.now()

    test_data = m.transform('test', 1)
    time3 = datetime.now()

    feature, view, seq_type, label = test_data

    # np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_feature_TestSysGait.npy", feature)
    # np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_view_TestSysGait.npy", view)
    # np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_seqtype_TestSysGait.npy", seq_type)
    # np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_label_TestSysGait.npy", label)
    
    np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_ori_feature_TestSysGait.npy", feature)
    np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_ori_view_TestSysGait.npy", view)
    np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_ori_seqtype_TestSysGait.npy", seq_type)
    np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_ori_label_TestSysGait.npy", label)

    print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))
    print("--!!--features saved")
    print("model init and load: {}, feature extrac:{}".format(time2-time1, time3-time2))
    
# get_npys_231224()

def get_npys_231224_for_noise():
    
    conf = {
        "WORK_PATH": "/home/nsec0/zyx/GaitSet-master/work",
        "CUDA_VISIBLE_DEVICES": "3",  # "0,1,2,3",
        "data": {
            # 'dataset_path': "/home/nsec0/zyx/GaitSet-master/pretreat_VR5Gait",
            'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait_noise2",
            'resolution': '64',
            'dataset': 'TestSysGaitNoise',# 'TestSysGait4',  # 'SysGait',  # 'CASIA-B', # VRGait # VR2Gait # 'VR5Gait'
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
            'total_iter': 150000,  #80000,  # 150000 for VR5Gait
            'margin': 0.2,
            'num_workers': 8,  # 3,
            'frame_num': 30,  # [NOTE] default:30, change here for factor 
            'model_name': 'SysGait4',  # 'SysGait4'
        },
    }
    
    for i in range(11):
        
        conf["data"]['dataset_path'] = "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait_noise/{:.1f}".format(i/10)
        
        time1 = datetime.now()
        m = initialization(conf, test=False)[0]
        m.load(80000)

        time2 = datetime.now()

        test_data = m.transform('test', 1)
        time3 = datetime.now()

        feature, view, seq_type, label = test_data

        np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_feature_TestSysGait_noise_{:0>2d}.npy".format(i), feature)
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_view_TestSysGait_noise_{:0>2d}.npy".format(i), view)
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_seqtype_TestSysGait_noise_{:0>2d}.npy".format(i), seq_type)
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new/infer_label_TestSysGait_noise_{:0>2d}.npy".format(i), label)

        print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))
        print("--!!--features saved")
        print("model init and load: {}, feature extrac:{}".format(time2-time1, time3-time2))

# get_npys_231224_for_noise()


def get_npys_240107_for_noise2():
    
    conf = {
        "WORK_PATH": "/home/nsec0/zyx/GaitSet-master/work",
        "CUDA_VISIBLE_DEVICES": "3",  # "0,1,2,3",
        "data": {
            # 'dataset_path': "/home/nsec0/zyx/GaitSet-master/pretreat_VR5Gait",
            'dataset_path': "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait_noise2",
            'resolution': '64',
            'dataset': 'TestSysGaitNoise',# 'TestSysGait4',  # 'SysGait',  # 'CASIA-B', # VRGait # VR2Gait # 'VR5Gait'
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
            'total_iter': 150000,  #80000,  # 150000 for VR5Gait
            'margin': 0.2,
            'num_workers': 8,  # 3,
            'frame_num': 30,  # [NOTE] default:30, change here for factor 
            'model_name': 'SysGait4',  # 'SysGait4'
        },
    }
    
    dataset_dir = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1',
                   '2', '4', '6', '8', '10', '12', '14', '16']
    
    for i in range(len(dataset_dir)):
        
        conf["data"]['dataset_path'] = "/home/nsec0/zyx/GaitSet-master/data_pretreat/pretreat_TestSysGait_noise2/{}".format(dataset_dir[i])
        
        time1 = datetime.now()
        m = initialization(conf, test=False)[0]
        m.load(80000)

        time2 = datetime.now()

        test_data = m.transform('test', 1)
        time3 = datetime.now()

        feature, view, seq_type, label = test_data

        # np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_feature_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), feature)
        # np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_view_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), view)
        # np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_seqtype_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), seq_type)
        # np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_label_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), label)
        
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_ori_feature_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), feature)
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_ori_view_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), view)
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_ori_seqtype_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), seq_type)
        np.save("/home/nsec0/zyx/GaitSet-master/npy_new2/infer_ori_label_TestSysGait_noise2_{}.npy".format(dataset_dir[i]), label)

        print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))
        print("--!!--features saved")
        print("model init and load: {}, feature extrac:{}".format(time2-time1, time3-time2))
        
get_npys_240107_for_noise2()    