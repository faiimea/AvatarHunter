from datetime import datetime
import numpy as np
import argparse
import os
from model.initialization import initialization
from model.utils import *
from config import conf
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

suffix = "_FBRL_Frame30"

time1 = datetime.now()
m = initialization(conf, test=False)[0]
m.load(80000)

time2 = datetime.now()

test_data = m.transform('test', 1)
time3 = datetime.now()

feature, view, seq_type, label = test_data

np.save("/home/nsec0/zyx/GaitSet-master/infer_ori_all_feature_VR5Gait{}.npy".format(suffix), feature)
np.save("/home/nsec0/zyx/GaitSet-master/infer_ori_all_view_VR5Gait{}.npy".format(suffix), view)
np.save("/home/nsec0/zyx/GaitSet-master/infer_ori_all_seqtype_VR5Gait{}.npy".format(suffix), seq_type)
np.save("/home/nsec0/zyx/GaitSet-master/infer_ori_all_label_VR5Gait{}.npy".format(suffix), label)

print("feature size: {}, view size {}, seq_type size {}, label size {}".format(feature.shape, len(view), len(seq_type), len(label)))
print("--!!--features saved")
print("model init and load: {}, feature extrac(for 10vol*10mh*10seq*4view):{}".format(time2-time1, time3-time2))


label = np.array(label)
label_list = list(set(label))

view = np.array(view)
view_list = list(set(view))

print("label_list: {}".format(label_list))
print("view list: {}".format(view_list))

# encode ps as mh*10 + seq ; pl as vol num; 
dists = np.zeros((100, 11, 100, 11))

seq_dict = []
for dict_mh_num in range(10):
    for dict_seq_num in range(10):
        seq_dict.append("mh{}-seq{}".format(dict_mh_num, dict_seq_num))


for probe_seq in seq_dict:
    for probe_label in label_list:
        for gallery_seq in seq_dict:
            for gallery_label in label_list:
                gallery_mask = np.isin(seq_type, gallery_seq) & np.isin(label, gallery_label)
                gallery_feature = feature[gallery_mask, :]

                probe_mask = np.isin(seq_type, probe_seq) & np.isin(label, probe_label)
                probe_feature = feature[probe_mask, :]

                dist = np.mean(cuda_dist(probe_feature, gallery_feature).cpu().numpy())
                        
                # ignore original ps, pl, gs, gl, rewrite them for convient encode / decode of dists
                ps = int(probe_seq.split('-')[0][-1]) * 10 + int(probe_seq.split('-')[1][-1])
                pl = int(probe_label[3:])
                gs = int(gallery_seq.split('-')[0][-1]) * 10 + int(gallery_seq.split('-')[1][-1])
                gl = int(gallery_label[3:])
                dists[ps, pl, gs, gl] = dist

time4 = datetime.now()
print("model init and load: {}, feature extrac(for 10vol*10mh*10seq*4view):{}, dists cal(for C_10vol*10mh*10seq^2):{}".format(time2-time1, time3-time2, time4-time3))

np.save("/home/nsec0/zyx/GaitSet-master/infer_ori_dists_VR5Gait{}.npy".format(suffix), dists)
print("--!!-- dists saved")
