import torch
import torch.nn.functional as F
import numpy as np
import os


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    # print("--Debug-- dist shape {}".format(dist.shape))

    return dist

# direction = "RL"

# def evaluation_new(data, config):
#     dataset = config['dataset'].split('-')[0]
#     feature, view, seq_type, label = data
    
#     print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))

#     label = np.array(label)
#     label_list = list(set(label))

#     view = np.array(view)
#     view_list = list(set(view))

#     print("label_list: {}".format(label_list))
#     print("view list: {}".format(view_list))
#     print("========================================================================\n")
    
#     probe_seq_dict = {'CASIA': [['nm-0s5', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
#                       'OUMVLP': [['00']],
#                       'SysGaitN': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
#                       'SysGait': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
#                       'TestSysGait': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
#                       'VRGait': ['balance-1', 'fat-1', 'high-1', 'strong-1'],
#                       'VR2Gait': ['boy-3', 'boy-4', 'robot-3', 'robot-4'],
#                       'VR3Gait': ['mh0-5', 'mh0-6', 'mh1-5', 'mh1-6', 'mh2-5', 'mh2-6'],
#                       'SysGait2': ['0-0', '1-0', '2-0'],
#                       'SysGait4': ['0-0', '1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '7-0', '8-0', '9-0'],
#                       'VR5Gait': ['mh0-seq0', 'mh1-seq0', 'mh2-seq0', 'mh3-seq0', 'mh4-seq0', 'mh5-seq0', 'mh6-seq0', 'mh7-seq0', 'mh8-seq0', 'mh9-seq0', 'mh0-seq1', 'mh1-seq1', 'mh2-seq1', 'mh3-seq1', 'mh4-seq1', 'mh5-seq1', 'mh6-seq1', 'mh7-seq1', 'mh8-seq1', 'mh9-seq1', 'mh0-seq2', 'mh1-seq2', 'mh2-seq2', 'mh3-seq2', 'mh4-seq2', 'mh5-seq2', 'mh6-seq2', 'mh7-seq2', 'mh8-seq2', 'mh9-seq2', 'mh0-seq3', 'mh1-seq3', 'mh2-seq3', 'mh3-seq3', 'mh4-seq3', 'mh5-seq3', 'mh6-seq3', 'mh7-seq3', 'mh8-seq3', 'mh9-seq3', 'mh0-seq4', 'mh1-seq4', 'mh2-seq4', 'mh3-seq4', 'mh4-seq4', 'mh5-seq4', 'mh6-seq4', 'mh7-seq4', 'mh8-seq4', 'mh9-seq4', 'mh0-seq5', 'mh1-seq5', 'mh2-seq5', 'mh3-seq5', 'mh4-seq5', 'mh5-seq5', 'mh6-seq5', 'mh7-seq5', 'mh8-seq5', 'mh9-seq5', 'mh0-seq6', 'mh1-seq6', 'mh2-seq6', 'mh3-seq6', 'mh4-seq6', 'mh5-seq6', 'mh6-seq6', 'mh7-seq6', 'mh8-seq6', 'mh9-seq6']}
#     gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
#                         'OUMVLP': [['01']],
#                         'SysGaitN': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
#                         'SysGait': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
#                         'TestSysGait': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
#                         'VRGait': ['balance-2', 'fat-2', 'high-2', 'strong-2'],
#                         'VR2Gait': ['boy-0', 'boy-1', 'boy-2', 'robot-0', 'robot-1', 'robot-2'],
#                         'VR3Gait': ['mh0-1', 'mh0-2', 'mh0-3', 'mh0-4', 'mh1-1', 'mh1-2', 'mh1-3', 'mh1-4', 'mh2-1', 'mh2-2', 'mh2-3', 'mh2-4'],
#                         'SysGait2': ['0-1', '1-1', '2-1'],
#                         'SysGait4': ['0-1', '1-1', '2-1', '3-1', '4-1', '5-1', '6-1', '7-1', '8-1', '9-1'],
#                         'VR5Gait': ['mh0-seq7', 'mh1-seq7', 'mh2-seq7', 'mh3-seq7', 'mh4-seq7', 'mh5-seq7', 'mh6-seq7', 'mh7-seq7', 'mh8-seq7', 'mh9-seq7', 'mh0-seq8', 'mh1-seq8', 'mh2-seq8', 'mh3-seq8', 'mh4-seq8', 'mh5-seq8', 'mh6-seq8', 'mh7-seq8', 'mh8-seq8', 'mh9-seq8', 'mh0-seq9', 'mh1-seq9', 'mh2-seq9', 'mh3-seq9', 'mh4-seq9', 'mh5-seq9', 'mh6-seq9', 'mh7-seq9', 'mh8-seq9', 'mh9-seq9']}

#     # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_VR5Gait_zsyModified_{}_frame30.npy"
#     # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_SysGaitN.npy"
#     # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_VR5Gait.npy"
#     # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_SysGait4.npy"
#     # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_TestSysGait.npy"
    
#     # if not os.path.exists(setting):
#     if True:

#         dists = np.zeros((len(probe_seq_dict[dataset]), len(label_list), len(gallery_seq_dict[dataset]), len(label_list)))

#         print(dists.shape)
#         # encode ps as mh*10 + seq ; pl as vol num; 
#         # dists = np.zeros((100, 11, 100, 11))

#         for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
#             for pl, probe_label in enumerate(label_list):
#                 for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
#                     for gl, gallery_label in enumerate(label_list):
#                         gallery_mask = np.isin(seq_type, gallery_seq) & np.isin(label, gallery_label)
#                         gallery_feature = feature[gallery_mask, :]

#                         probe_mask = np.isin(seq_type, probe_seq) & np.isin(label, probe_label)
#                         probe_feature = feature[probe_mask, :]

#                         dist = np.mean(cuda_dist(probe_feature, gallery_feature).cpu().numpy())
                        
#                         # ignore original ps, pl, gs, gl, rewrite them for convient encode / decode of dists
#                         # ps = int(probe_seq.split('-')[0][-1]) * 10 + int(probe_seq.split('-')[1][-1])
#                         # pl = int(probe_label[3:])
#                         # gs = int(gallery_seq.split('-')[0][-1]) * 10 + int(gallery_seq.split('-')[1][-1])
#                         # gl = int(gallery_label[3:])
#                         dists[ps, pl, gs, gl] = dist

#         np.save(setting, dists)
#         print("--!!--saved")

#     else:
#         # dists = np.load("/home/nsec0/zyx/GaitSet-master/infer_dists_VR5Gait_128.npy") by zsy
#         dists = np.load(setting)
#         print("--!!--loaded")

#     # return 
        

#     infer_cross_right = 0
#     infer_cross_time = 0
    
#     infer_inner_right = 0
#     infer_inner_time = 0

#     IA_threshold = 3  # len(gallery_seq_dict[dataset]) / (num_mh)
#     # CA_threshold = len(label_list) * len(gallery_seq_dict[dataset]) // 10  # top 20% of gallery

#     CA_threshold = 3

#     for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
#         for pl, probe_label in enumerate(label_list):  # for every probe (label + seq)

#             print("distances to {} {}".format(probe_seq, probe_label))
#             IA_seq = []
#             IA_label = []
#             IA_dist = []
            
#             CA_seq = []
#             CA_label = []
#             CA_dist = []
            
#             infer_inner_time += 1
#             infer_cross_time += 1
            
#             for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
#                 if gallery_seq.split('-')[0]  == probe_seq.split('-')[0]:
#                     for gl, gallery_label in enumerate(label_list):
#                         IA_seq.append(gallery_seq)
#                         IA_label.append(gallery_label)
#                         IA_dist.append(dists[ps, pl, gs, gl])
#                 else:
#                     for gl, gallery_label in enumerate(label_list):
#                         CA_seq.append(gallery_seq)
#                         CA_label.append(gallery_label)
#                         CA_dist.append(dists[ps, pl, gs, gl])
                    
#             IA_sort = np.argsort(IA_dist)
#             sorted_seq = [IA_seq[idx] for idx in IA_sort]
#             sorted_label = [IA_label[idx] for idx in IA_sort]
#             # print("--Debug-- sorted_seq {}, sorted_label {}".format(sorted_seq, sorted_label))

#             sorted_label_set = []
#             for label in sorted_label:
#                 if label not in sorted_label_set:
#                     sorted_label_set.append(label)

#             print("IA sorted_label: {}, IA sorted_label_set: {}".format(sorted_label[:30], sorted_label_set))

#             # get the label that appears the most times
#             # if max(sorted_label, key=sorted_label.count) == probe_label:
#             #     infer_inner_right += 1
#             # else:
#             #      print("[LOG] IA failed")
                
#             # check top-threshold acc
#             if probe_label in sorted_label_set[:IA_threshold]:
#                infer_inner_right += 1
#                print('---------IA plus 1---------')
#             else:
#                 print("[LOG] IA failed")
                    
#             CA_sort = np.argsort(CA_dist)
#             sorted_seq = [CA_seq[idx] for idx in CA_sort]
#             sorted_label = [CA_label[idx] for idx in CA_sort]
#             # print("--Debug-- sorted_seq {}, sorted_label {}".format(sorted_seq, sorted_label))

#             sorted_label_set = []
#             for label in sorted_label:
#                 if label not in sorted_label_set:
#                     sorted_label_set.append(label)

#             print("CA sorted_label: {}, CA sorted_label_set: {}".format(sorted_label[:30], sorted_label_set))

#             # get the label that appears the most times
#             # if max(sorted_label, key=sorted_label.count) == probe_label:
#             #     infer_cross_right += 1
#             # else:
#             #      print("[LOG] CA failed")

#             # check top-threshold acc
#             if probe_label in sorted_label_set[:CA_threshold]:
#                 infer_cross_right += 1
#                 print('----------CA plus 1----------')
#             else:
#                 print("[LOG] CA failed")

#             print("=======================================================\n")

#     print("acc(cross avatar) = {} / {} = {}".format(infer_cross_right, infer_cross_time, infer_cross_right / infer_cross_time))

#     print("acc(only identical avatar) = {} / {} = {}".format(infer_inner_right, infer_inner_time, infer_inner_right / infer_inner_time))
    
#     print("acc(include identical avatar) = {} / {} = {}".format(infer_inner_right+infer_cross_right, infer_inner_time+infer_cross_time, (infer_inner_right+infer_cross_right) / (infer_inner_time + infer_cross_time)))


def evaluation_new(data, config, setting):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    
    print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))

    label = np.array(label)
    label_list = list(set(label))

    view = np.array(view)
    view_list = list(set(view))

    print("label_list: {}".format(label_list))
    print("view list: {}".format(view_list))
    print("========================================================================\n")
    
    probe_seq_dict = {'CASIA': [['nm-0s5', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'TestSysGaitN': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
                      'SysGait': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
                      'TestSysGait': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
                      'VRGait': ['balance-1', 'fat-1', 'high-1', 'strong-1'],
                      'VR2Gait': ['boy-3', 'boy-4', 'robot-3', 'robot-4'],
                      'VR3Gait': ['mh0-5', 'mh0-6', 'mh1-5', 'mh1-6', 'mh2-5', 'mh2-6'],
                      'SysGait2': ['0-0', '1-0', '2-0'],
                      'SysGait4': ['0-0', '1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '7-0', '8-0', '9-0'],
                      'VR5Gait': ['mh0-seq0', 'mh1-seq0', 'mh2-seq0', 'mh3-seq0', 'mh4-seq0', 'mh5-seq0', 'mh6-seq0', 'mh7-seq0', 'mh8-seq0', 'mh9-seq0', 'mh0-seq1', 'mh1-seq1', 'mh2-seq1', 'mh3-seq1', 'mh4-seq1', 'mh5-seq1', 'mh6-seq1', 'mh7-seq1', 'mh8-seq1', 'mh9-seq1', 'mh0-seq2', 'mh1-seq2', 'mh2-seq2', 'mh3-seq2', 'mh4-seq2', 'mh5-seq2', 'mh6-seq2', 'mh7-seq2', 'mh8-seq2', 'mh9-seq2', 'mh0-seq3', 'mh1-seq3', 'mh2-seq3', 'mh3-seq3', 'mh4-seq3', 'mh5-seq3', 'mh6-seq3', 'mh7-seq3', 'mh8-seq3', 'mh9-seq3', 'mh0-seq4', 'mh1-seq4', 'mh2-seq4', 'mh3-seq4', 'mh4-seq4', 'mh5-seq4', 'mh6-seq4', 'mh7-seq4', 'mh8-seq4', 'mh9-seq4', 'mh0-seq5', 'mh1-seq5', 'mh2-seq5', 'mh3-seq5', 'mh4-seq5', 'mh5-seq5', 'mh6-seq5', 'mh7-seq5', 'mh8-seq5', 'mh9-seq5', 'mh0-seq6', 'mh1-seq6', 'mh2-seq6', 'mh3-seq6', 'mh4-seq6', 'mh5-seq6', 'mh6-seq6', 'mh7-seq6', 'mh8-seq6', 'mh9-seq6']}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'TestSysGaitN': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
                        'SysGait': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
                        'TestSysGait': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
                        'VRGait': ['balance-2', 'fat-2', 'high-2', 'strong-2'],
                        'VR2Gait': ['boy-0', 'boy-1', 'boy-2', 'robot-0', 'robot-1', 'robot-2'],
                        'VR3Gait': ['mh0-1', 'mh0-2', 'mh0-3', 'mh0-4', 'mh1-1', 'mh1-2', 'mh1-3', 'mh1-4', 'mh2-1', 'mh2-2', 'mh2-3', 'mh2-4'],
                        'SysGait2': ['0-1', '1-1', '2-1'],
                        'SysGait4': ['0-1', '1-1', '2-1', '3-1', '4-1', '5-1', '6-1', '7-1', '8-1', '9-1'],
                        'VR5Gait': ['mh0-seq7', 'mh1-seq7', 'mh2-seq7', 'mh3-seq7', 'mh4-seq7', 'mh5-seq7', 'mh6-seq7', 'mh7-seq7', 'mh8-seq7', 'mh9-seq7', 'mh0-seq8', 'mh1-seq8', 'mh2-seq8', 'mh3-seq8', 'mh4-seq8', 'mh5-seq8', 'mh6-seq8', 'mh7-seq8', 'mh8-seq8', 'mh9-seq8', 'mh0-seq9', 'mh1-seq9', 'mh2-seq9', 'mh3-seq9', 'mh4-seq9', 'mh5-seq9', 'mh6-seq9', 'mh7-seq9', 'mh8-seq9', 'mh9-seq9']}

    # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_VR5Gait_zsyModified_{}_frame30.npy"
    # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_SysGaitN.npy"
    # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_VR5Gait.npy"
    # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_SysGait4.npy"
    # setting = "/home/nsec0/zyx/GaitSet-master/npy_dist/infer_dists_TestSysGait.npy"
    
    # if not os.path.exists(setting):
    if True:

        dists = np.zeros((len(probe_seq_dict[dataset]), len(label_list), len(gallery_seq_dict[dataset]), len(label_list)))

        print(dists.shape)
        # encode ps as mh*10 + seq ; pl as vol num; 
        # dists = np.zeros((100, 11, 100, 11))

        for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
            for pl, probe_label in enumerate(label_list):
                for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
                    for gl, gallery_label in enumerate(label_list):
                        gallery_mask = np.isin(seq_type, gallery_seq) & np.isin(label, gallery_label)
                        gallery_feature = feature[gallery_mask, :]

                        probe_mask = np.isin(seq_type, probe_seq) & np.isin(label, probe_label)
                        probe_feature = feature[probe_mask, :]

                        dist = np.mean(cuda_dist(probe_feature, gallery_feature).cpu().numpy())
                        
                        # ignore original ps, pl, gs, gl, rewrite them for convient encode / decode of dists
                        # [Note] This encoding is designed for ReSortImgs/evaluate_*.py
                        # ps = int(probe_seq.split('-')[0][-1]) * 10 + int(probe_seq.split('-')[1][-1])
                        # pl = int(probe_label[3:])
                        # gs = int(gallery_seq.split('-')[0][-1]) * 10 + int(gallery_seq.split('-')[1][-1])
                        # gl = int(gallery_label[3:])
                        dists[ps, pl, gs, gl] = dist

        np.save(setting, dists)
        print("--!!--saved")

    else:
        # dists = np.load("/home/nsec0/zyx/GaitSet-master/infer_dists_VR5Gait_128.npy") by zsy
        dists = np.load(setting)
        print("--!!--loaded")

    # return 
        

    infer_cross_right = 0
    infer_cross_time = 0
    
    infer_inner_right = 0
    infer_inner_time = 0

    IA_threshold = 2  # len(gallery_seq_dict[dataset]) / (num_mh)
    # CA_threshold = len(label_list) * len(gallery_seq_dict[dataset]) // 10  # top 20% of gallery

    CA_threshold = 2

    for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
        for pl, probe_label in enumerate(label_list):  # for every probe (label + seq)

            print("distances to {} {}".format(probe_seq, probe_label))
            IA_seq = []
            IA_label = []
            IA_dist = []
            
            CA_seq = []
            CA_label = []
            CA_dist = []
            
            infer_inner_time += 1
            infer_cross_time += 1
            
            for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
                if gallery_seq.split('-')[0]  == probe_seq.split('-')[0]:
                    for gl, gallery_label in enumerate(label_list):
                        IA_seq.append(gallery_seq)
                        IA_label.append(gallery_label)
                        IA_dist.append(dists[ps, pl, gs, gl])
                else:
                    for gl, gallery_label in enumerate(label_list):
                        CA_seq.append(gallery_seq)
                        CA_label.append(gallery_label)
                        CA_dist.append(dists[ps, pl, gs, gl])
                    
            IA_sort = np.argsort(IA_dist)
            sorted_seq = [IA_seq[idx] for idx in IA_sort]
            sorted_label = [IA_label[idx] for idx in IA_sort]
            # print("--Debug-- sorted_seq {}, sorted_label {}".format(sorted_seq, sorted_label))

            sorted_label_set = []
            for label in sorted_label:
                if label not in sorted_label_set:
                    sorted_label_set.append(label)

            print("IA sorted_label: {}, IA sorted_label_set: {}".format(sorted_label[:30], sorted_label_set))

            # get the label that appears the most times
            # if max(sorted_label, key=sorted_label.count) == probe_label:
            #     infer_inner_right += 1
            # else:
            #      print("[LOG] IA failed")
                
            # check top-threshold acc
            if probe_label in sorted_label_set[:IA_threshold]:
               infer_inner_right += 1
               print('---------IA plus 1---------')
            else:
                print("[LOG] IA failed")
                    
            CA_sort = np.argsort(CA_dist)
            sorted_seq = [CA_seq[idx] for idx in CA_sort]
            sorted_label = [CA_label[idx] for idx in CA_sort]
            # print("--Debug-- sorted_seq {}, sorted_label {}".format(sorted_seq, sorted_label))

            sorted_label_set = []
            for label in sorted_label:
                if label not in sorted_label_set:
                    sorted_label_set.append(label)

            print("CA sorted_label: {}, CA sorted_label_set: {}".format(sorted_label[:30], sorted_label_set))

            # get the label that appears the most times
            # if max(sorted_label, key=sorted_label.count) == probe_label:
            #     infer_cross_right += 1
            # else:
            #      print("[LOG] CA failed")

            # check top-threshold acc
            if probe_label in sorted_label_set[:CA_threshold]:
                infer_cross_right += 1
                print('----------CA plus 1----------')
            else:
                print("[LOG] CA failed")

            print("=======================================================\n")

    print("acc(cross avatar) = {} / {} = {}".format(infer_cross_right, infer_cross_time, infer_cross_right / infer_cross_time))

    print("acc(only identical avatar) = {} / {} = {}".format(infer_inner_right, infer_inner_time, infer_inner_right / infer_inner_time))
    
    print("acc(include identical avatar) = {} / {} = {}".format(infer_inner_right+infer_cross_right, infer_inner_time+infer_cross_time, (infer_inner_right+infer_cross_right) / (infer_inner_time + infer_cross_time)))




def evaluation_cross_avatar(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data

    print("--Debug-- shape feature={} , view={}, seq_type={}, label={}".format(feature.shape, len(view), len(seq_type), len(label)))

    label = np.array(label)
    label_list = list(set(label))

    view = np.array(view)
    view_list = list(set(view))

    probe_seq_dict = {'CASIA': [['nm-0s5', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'TestSysGait': ['balanced-00', 'child-00', 'fat-00', 'strong-00', 'tall-00'],
                      'VRGait': ['balance-1', 'fat-1', 'high-1', 'strong-1'],
                      'VR2Gait': ['boy-3', 'boy-4', 'robot-3', 'robot-4'],
                      'VR3Gait': ['mh0-5', 'mh0-6', 'mh1-5', 'mh1-6', 'mh2-5', 'mh2-6'],
                      'SysGait2': ['0-0', '1-0', '2-0'],
                      'SysGait4': ['0-0', '1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '7-0', '8-0', '9-0'],
                      'VR4Gait': ['mh0-seq0', 'mh1-seq0', 'mh2-seq0', 'mh3-seq0', 'mh5-seq0', 'mh6-seq0', 'mh7-seq0', 'mh8-seq0', 'mh9-seq0', 'mh0-seq1', 'mh1-seq1', 'mh2-seq1', 'mh3-seq1', 'mh5-seq1', 'mh6-seq1', 'mh7-seq1', 'mh8-seq1', 'mh9-seq1', 'mh0-seq2', 'mh1-seq2', 'mh2-seq2', 'mh3-seq2', 'mh5-seq2', 'mh6-seq2', 'mh7-seq2', 'mh8-seq2', 'mh9-seq2', 'mh0-seq3', 'mh1-seq3', 'mh2-seq3', 'mh3-seq3', 'mh5-seq3', 'mh6-seq3', 'mh7-seq3', 'mh8-seq3', 'mh9-seq3', 'mh0-seq4', 'mh1-seq4', 'mh2-seq4', 'mh3-seq4', 'mh5-seq4', 'mh6-seq4', 'mh7-seq4', 'mh8-seq4', 'mh9-seq4', 'mh0-seq5', 'mh1-seq5', 'mh2-seq5', 'mh3-seq5', 'mh5-seq5', 'mh6-seq5', 'mh7-seq5', 'mh8-seq5', 'mh9-seq5', 'mh0-seq6', 'mh1-seq6', 'mh2-seq6', 'mh3-seq6', 'mh5-seq6', 'mh6-seq6', 'mh7-seq6', 'mh8-seq6', 'mh9-seq6'],
                      'VR5Gait': ['mh0-seq0', 'mh1-seq0', 'mh2-seq0', 'mh3-seq0', 'mh4-seq0', 'mh5-seq0', 'mh6-seq0', 'mh7-seq0', 'mh8-seq0', 'mh9-seq0', 'mh0-seq1', 'mh1-seq1', 'mh2-seq1', 'mh3-seq1', 'mh4-seq1', 'mh5-seq1', 'mh6-seq1', 'mh7-seq1', 'mh8-seq1', 'mh9-seq1', 'mh0-seq2', 'mh1-seq2', 'mh2-seq2', 'mh3-seq2', 'mh4-seq2', 'mh5-seq2', 'mh6-seq2', 'mh7-seq2', 'mh8-seq2', 'mh9-seq2', 'mh0-seq3', 'mh1-seq3', 'mh2-seq3', 'mh3-seq3', 'mh4-seq3', 'mh5-seq3', 'mh6-seq3', 'mh7-seq3', 'mh8-seq3', 'mh9-seq3', 'mh0-seq4', 'mh1-seq4', 'mh2-seq4', 'mh3-seq4', 'mh4-seq4', 'mh5-seq4', 'mh6-seq4', 'mh7-seq4', 'mh8-seq4', 'mh9-seq4', 'mh0-seq5', 'mh1-seq5', 'mh2-seq5', 'mh3-seq5', 'mh4-seq5', 'mh5-seq5', 'mh6-seq5', 'mh7-seq5', 'mh8-seq5', 'mh9-seq5', 'mh0-seq6', 'mh1-seq6', 'mh2-seq6', 'mh3-seq6', 'mh4-seq6', 'mh5-seq6', 'mh6-seq6', 'mh7-seq6', 'mh8-seq6', 'mh9-seq6']}
    
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'TestSysGait': ['balanced-01', 'child-01', 'fat-01', 'strong-01', 'tall-01'],
                        'VRGait': ['balance-2', 'fat-2', 'high-2', 'strong-2'],
                        'VR2Gait': ['boy-0', 'boy-1', 'boy-2', 'robot-0', 'robot-1', 'robot-2'],
                        'VR3Gait': ['mh0-1', 'mh0-2', 'mh0-3', 'mh0-4', 'mh1-1', 'mh1-2', 'mh1-3', 'mh1-4', 'mh2-1', 'mh2-2', 'mh2-3', 'mh2-4'],
                        'SysGait2': ['0-1', '1-1', '2-1'],
                        'SysGait4': ['0-1', '1-1', '2-1', '3-1', '4-1', '5-1', '6-1', '7-1', '8-1', '9-1'],
                        'VR4Gait': ['mh0-seq7', 'mh1-seq7', 'mh2-seq7', 'mh3-seq7', 'mh5-seq7', 'mh6-seq7', 'mh7-seq7', 'mh8-seq7', 'mh9-seq7', 'mh0-seq8', 'mh1-seq8', 'mh2-seq8', 'mh3-seq8', 'mh5-seq8', 'mh6-seq8', 'mh7-seq8', 'mh8-seq8', 'mh9-seq8', 'mh0-seq9', 'mh1-seq9', 'mh2-seq9', 'mh3-seq9', 'mh5-seq9', 'mh6-seq9', 'mh7-seq9', 'mh8-seq9', 'mh9-seq9'],
                        'VR5Gait': ['mh0-seq7', 'mh1-seq7', 'mh2-seq7', 'mh3-seq7', 'mh4-seq7', 'mh5-seq7', 'mh6-seq7', 'mh7-seq7', 'mh8-seq7', 'mh9-seq7', 'mh0-seq8', 'mh1-seq8', 'mh2-seq8', 'mh3-seq8', 'mh4-seq8', 'mh5-seq8', 'mh6-seq8', 'mh7-seq8', 'mh8-seq8', 'mh9-seq8', 'mh0-seq9', 'mh1-seq9', 'mh2-seq9', 'mh3-seq9', 'mh4-seq9', 'mh5-seq9', 'mh6-seq9', 'mh7-seq9', 'mh8-seq9', 'mh9-seq9']}

    """    
    ## export feature vector data

    feature_probe = np.zeros((len(label_list), len(probe_seq_dict[dataset]), len(view_list), feature.shape[1]))
    print("-debug- shape of feature_probe = {}".format(feature_probe.shape))

    for pl, probe_label in enumerate(label_list):
        for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
            for pv, probe_view in enumerate(view_list):
                probe_mask = np.isin(label, probe_label) & np.isin(seq_type, probe_seq) & np.isin(view, probe_view)
                probe_feature = feature[probe_mask, :]
                # print('-debug- pl={}, ps={}, pv={}, probe_feature={}'.format(probe_label, probe_seq, probe_view, probe_feature))
                feature_probe[pl, ps, pv, :] = probe_feature
    # np.save("/home/nsec0/zyx/GaitSet-master/infer_feature_probe_TestVR3Gait.npy", feature_probe)

    feature_gallery = np.zeros((len(label_list), len(gallery_seq_dict[dataset]), len(view_list), feature.shape[1]))
    for gl, gallery_label in enumerate(label_list):
        for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
            for gv, gallery_view in enumerate(view_list):
                gallery_mask = np.isin(label, gallery_label) & np.isin(seq_type, gallery_seq) & np.isin(view, gallery_view)
                gallery_feature = feature[gallery_mask, :]
                feature_gallery[gl, gs, gv, :] = gallery_feature
    # np.save("/home/nsec0/zyx/GaitSet-master/infer_feature_gallery_TestVR3Gait.npy", feature_gallery)
    """


    # if not os.path.exists("/home/nsec0/zyx/GaitSet-master/infer_dists_VR5Gait_zsyModified_{}_frame30.npy".format(direction)):
    if True:
        dists = np.zeros((len(probe_seq_dict[dataset]), len(label_list), len(gallery_seq_dict[dataset]), len(label_list)))
        for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
            for pl, probe_label in enumerate(label_list):
                for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
                    for gl, gallery_label in enumerate(label_list):
                        gallery_mask = np.isin(seq_type, gallery_seq) & np.isin(label, gallery_label)
                        gallery_feature = feature[gallery_mask, :]

                        probe_mask = np.isin(seq_type, probe_seq) & np.isin(label, probe_label)
                        probe_feature = feature[probe_mask, :]

                        dist = np.mean(cuda_dist(probe_feature, gallery_feature).cpu().numpy())
                        dists[ps, pl, gs, gl] = dist

        # np.save("/home/nsec0/zyx/GaitSet-master/infer_dists_VR5Gait_zsyModified_{}_frame30.npy".format(direction), dists)
        print("--!!--saved")

    else:
        dists = np.load("/home/nsec0/zyx/GaitSet-master/infer_dists_VR5Gait_zsyModified_{}_frame30.npy".format(direction))
        print("--!!--loaded")

    infer_cross_right = 0
    infer_cross_time = 0
    
    infer_inner_right = 0
    infer_inner_time = 0
    
    IA_threshold = 3

    CA_threshold = 3
    
    print("label_list: {}".format(label_list))
    print("view list: {}".format(view_list))
    print("========================================================================\n")

    for ps, probe_seq in enumerate(probe_seq_dict[dataset]):
        for pl, probe_label in enumerate(label_list):
            
            print("distances to {} {}".format(probe_seq, probe_label))
            #print(label_list)
            for gs, gallery_seq in enumerate(gallery_seq_dict[dataset]):
                
                if gallery_seq.split('-')[0]  == probe_seq.split('-')[0]:
                    infer_inner_time += 1
                    idx = np.argsort(dists[ps, pl, gs, :])
                    
                    print("=======gallery_seq={}".format(gallery_seq))
                    print("the most like are: {}".format([label_list[i] for i in idx]))
                    most_likely = [label_list[i] for i in idx]
                    
                    if probe_label in most_likely[:IA_threshold]:
                        infer_inner_right += 1
                        print("--IA Succ!--")
                    else:
                        print("--IA Failed!--")
                        
                    # if label_list[idx[0]] == probe_label:
                    #     infer_inner_right += 1
                    #     print("--IA Succ!--")
                    # else:
                    #     print("--IA Failed!--")
                    
                else:
                    infer_cross_time += 1
                    idx = np.argsort(dists[ps, pl, gs, :])
                    
                    print("========gallery_seq={}".format(gallery_seq))
                    print("the most like are: {}".format([label_list[i] for i in idx]))
                    most_likely = [label_list[i] for i in idx]
                    
                    # if label_list[idx[0]] == probe_label:
                    #     infer_cross_right += 1
                    #     print("--CA Succ!--")
                    # else:
                    #     print("--CA Failed!--")
                    
                    if probe_label in most_likely[:CA_threshold]:
                        infer_cross_right += 1
                        print("--CA Succ!--")
                    else:
                        print("--CA Failed!--")
                    
                print("specific distances are: ", gallery_seq, dists[ps, pl, gs, :])

            print("=======================================================\n")
    print("acc(cross avatar) = {} / {} = {}".format(infer_cross_right, infer_cross_time, infer_cross_right / infer_cross_time))

    print("acc(only identical avatar) = {} / {} = {}".format(infer_inner_right, infer_inner_time, infer_inner_right / infer_inner_time))
    
    print("acc(include identical avatar) = {} / {} = {}".format(infer_inner_right+infer_cross_right, infer_inner_time+infer_cross_time, (infer_inner_right+infer_cross_right) / (infer_inner_time + infer_cross_time)))
    

def evaluation_avatar(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data

    label = np.array(label)
    label_list = list(set(label))

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'SysGait': [['seq-00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'SysGait': [['seq-01']]}

    infer_right = 0
    infer_num = 0
    for probe_seq in probe_seq_dict[dataset]:
        for probe_label in label_list:
            dists = {}
            infer_num += 1
            for gallery_label in label_list:
                dist_within_gseq = []
                for gallery_seq in gallery_seq_dict[dataset]:
                    gallery_mask = np.isin(seq_type, gallery_seq) & np.isin(label, gallery_label)
                    gallery_feature = feature[gallery_mask, :]

                    probe_mask = np.isin(seq_type, probe_seq) & np.isin(label, probe_label)
                    probe_feature = feature[probe_mask, :]

                    dist = np.mean(cuda_dist(probe_feature, gallery_feature).cpu().numpy())
                    dist_within_gseq.append(dist)
                dists[gallery_label] = min(dist_within_gseq)
            #print("distances to {}: {}".format(probe_label, dists))
            new_dic = sorted(dists.items(), key=lambda d: d[1], reverse=False)
            print("sorted distances to {}: {}\n".format(probe_label, new_dic))
            if probe_label == new_dic[0][0]:
                infer_right += 1
    print("acc = {} / {} = {}".format(infer_right, infer_num, infer_right / infer_num))


def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'SysGait': [['seq-00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'SysGait': [['seq-01']]}

    num_rank = 1
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0)
                        * 100 / dist.shape[0], 2)

    return acc
