from datetime import datetime
import numpy as np
import argparse
import os
from model.initialization import initialization
from model.utils import *
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=opt.cache)[0]

print(conf)

# load model checkpoint of iteration opt.iter
#print('Loading the model of iteration %d...' % opt.iter)

# [NOTE] here can be used to compare performance of different iters of TrainSysGait4
m.load(opt.iter)
#print('Transforming...')
time = datetime.now()


setting = "/home/nsec0/zyx/GaitSet-master/infer_all_{}_TestSysGait_noise_00.npy"
# setting = "/home/nsec0/zyx/GaitSet-master/infer_ori_all_{}_VR5Gait.npy"
# setting = "/home/nsec0/zyx/GaitSet-master/infer_ori_all_{}_SysGait4.npy"
# setting = "/home/nsec0/zyx/GaitSet-master/infer_ori_all_{}_TestSysGait.npy"

# if not os.path.exists(setting.format('feature')):
if True:
    test = m.transform('test', opt.batch_size)
    feature, view, seq_type, label = test
    np.save(setting.format('feature'), feature)
    np.save(setting.format('view'), view)
    np.save(setting.format('seq_type'), seq_type)
    np.save(setting.format('label'), label)
else:
    feature = np.load(setting.format('feature'))
    view = np.load(setting.format('view'))
    seq_type = np.load(setting.format('seq_type'))
    label = np.load(setting.format('label'))
    test = feature, view, seq_type, label


# For R
# if not os.path.exists("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait_R.npy"):
#     test = m.transform('test', opt.batch_size)
#     feature, view, seq_type, label = test
#     np.save("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait_R.npy", feature)
#     np.save("/home/nsec0/zyx/GaitSet-master/infer_all_view_VR5Gait_R.npy", view)
#     np.save("/home/nsec0/zyx/GaitSet-master/infer_all_seqtype_VR5Gait_R.npy", seq_type)
#     np.save("/home/nsec0/zyx/GaitSet-master/infer_all_label_VR5Gait_R.npy", label)
# else:
#     # feature = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait.npy")
#     # view = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_view_VR5Gait.npy")
#     # seq_type = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_seqtype_VR5Gait.npy")
#     # label = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_label_VR5Gait.npy")
#     # test = feature, view, seq_type, label 
#     # by zsy
#     feature = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait_R.npy")
#     view = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_view_VR5Gait_R.npy")
#     seq_type = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_seqtype_VR5Gait_R.npy")
#     label = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_label_VR5Gait_R.npy")
#     test = feature, view, seq_type, label

    
# For RL_frame30 
# direction = "RL"
# if not os.path.exists("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait_zsyModified_{}_frame30.npy".format(direction)):
#     test = m.transform('test', opt.batch_size)
#     feature, view, seq_type, label = test
#     np.save("/home/nsec0/zyx/GaitSet-master/npy_feature/infer_all_feature_VR5Gait_zsyModified_{}_frame30.npy".format(direction), feature)
#     np.save("/home/nsec0/zyx/GaitSet-master/npy_view/infer_all_view_VR5Gait_zsyModified_{}_frame30.npy".format(direction), view)
#     np.save("/home/nsec0/zyx/GaitSet-master/npy_seqtype/infer_all_seqtype_VR5Gait_zsyModified_{}_frame30.npy".format(direction), seq_type)
#     np.save("/home/nsec0/zyx/GaitSet-master/npy_label/infer_all_label_VR5Gait_zsyModified_{}_frame30.npy".format(direction), label)
# else:
#     # feature = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_feature_VR5Gait.npy")
#     # view = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_view_VR5Gait.npy")
#     # seq_type = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_seqtype_VR5Gait.npy")
#     # label = np.load("/home/nsec0/zyx/GaitSet-master/infer_all_label_VR5Gait.npy")
#     # test = feature, view, seq_type, label 
#     # by zsy
#     feature = np.load("/home/nsec0/zyx/GaitSet-master/npy_feature/infer_all_feature_VR5Gait_zsyModified_{}_frame30.npy".format(direction))
#     view = np.load("/home/nsec0/zyx/GaitSet-master/npy_view/infer_all_view_VR5Gait_zsyModified_{}_frame30.npy".format(direction))
#     seq_type = np.load("/home/nsec0/zyx/GaitSet-master/npy_seqtype/infer_all_seqtype_VR5Gait_zsyModified_{}_frame30.npy".format(direction))
#     label = np.load("/home/nsec0/zyx/GaitSet-master/npy_label/infer_all_label_VR5Gait_zsyModified_{}_frame30.npy".format(direction))
#     test = feature, view, seq_type, label


#print('Evaluating...')


#print("New evaluation method: ...")
#evaluation_avatar(test, conf['data'])

#print("Cross-avatar evaluation...")
# evaluation_cross_avatar(test, conf['data'])
# evaluation_new(test, conf['data'])
evaluation_new(test, conf['data'], setting.format('dists'))


#acc = evaluation(test, conf['data'])
#print('Evaluation complete. Cost:', datetime.now() - time)

# Print rank-1 accuracy of the best model
# e.g.
# ===Rank-1 (Include identical-view cases)===
# NM: 95.405,     BG: 88.284,     CL: 72.041
#for i in range(1):
#    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
#    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
#        np.mean(acc[0, :, :, i]),
#        np.mean(acc[1, :, :, i]),
#        np.mean(acc[2, :, :, i])))


#for view in range(5):
#    print("==Rank-1 only identical-view cases== -> {}".format(np.mean(acc[0, view, view, 0])))
#print("==Rank-1 Include identical-view cases== -> {}".format(np.mean(acc[0, :, :, 0])))


# Print rank-1 accuracy of the best model，excluding identical-view cases
# e.g.
# ===Rank-1 (Exclude identical-view cases)===
# NM: 94.964,     BG: 87.239,     CL: 70.355
#for i in range(1):
#    print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
#    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
#        de_diag(acc[0, :, :, i]),
#        de_diag(acc[1, :, :, i]),
#        de_diag(acc[2, :, :, i])))


#print("==Rank-1 Exclude identical-view cases== -> {}".format(de_diag(acc[0, :, :, 0])))


# Print rank-1 accuracy of the best model (Each Angle)
# e.g.
# ===Rank-1 of each angle (Exclude identical-view cases)===
# NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
# BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
# CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]

#np.set_printoptions(precision=2, floatmode='fixed')

#for i in range(1):
#    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
#    print('NM:', de_diag(acc[0, :, :, i], True))
#    print('BG:', de_diag(acc[1, :, :, i], True))
#    print('CL:', de_diag(acc[2, :, :, i], True))

#print("==Rank-1 of each angle Exclude identical-view cases== -> {}".format(de_diag(acc[0, :, :, 0], True)))
