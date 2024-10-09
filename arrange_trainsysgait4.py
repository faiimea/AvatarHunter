import shutil
import os
import get_npys_auto


def arrange_TrainSysGait4_adding_new_avatar(mhs, before_or_after):
	original_root_path = "/home/nsec0/zyx/GaitSet-master/pretreat_TrainSysGait4"
	root_path = "/home/nsec0/zyx/GaitSet-master/pretreat_TrainSysGait4_{}".format(before_or_after)
	for vol_num in [7, 8, 9, 12, 15, 16, 26, 27, 29, 32, 35, 36, 38, 39, 40, 41, 47, 56]:
		for mh_num in mhs:
			for view in ['000', '072', '090', '144', '180', '216', '270', '288']:
				sub_dir = os.path.join(root_path, str(vol_num), str(mh_num), view)
				original_sub_dir = os.path.join(original_root_path, str(vol_num), str(mh_num), view)
				shutil.copytree(original_sub_dir, sub_dir)
				print("copy {} to {}".format(original_sub_dir, sub_dir))


def arrange_VR5Gait_for_views(views):
	original_root_path = "/home/nsec0/zyx/GaitSet-master/pretreat_VR5Gait"
	root_path = "/home/nsec0/zyx/GaitSet-master/pretreat_VR5Gait_{}".format(''.join(views))
	for vol_num in range(1, 11):
		for mh_num in range(10):
			for seq_num in range(10):
				for view in views:
					sub_dir = os.path.join(root_path, "vol{}".format(vol_num), "mh{}-seq{}".format(mh_num, seq_num), view)
					original_sub_dir = os.path.join(original_root_path, "vol{}".format(vol_num), "mh{}-seq{}".format(mh_num, seq_num), view)
					shutil.copytree(original_sub_dir, sub_dir)
					print("copy {} to {}".format(original_sub_dir, sub_dir))

# arrange_TrainSysGait4_adding_new_avatar([1, 7], 'before')
# arrange_TrainSysGait4_adding_new_avatar([1, 5, 7], 'after')

to_add_views = ['B', 'L', 'BR', 'BL', 'FL', 'BRL', 'FBL']

arrange_VR5Gait_for_views(views=['B', ])
arrange_VR5Gait_for_views(views=['L', ])
arrange_VR5Gait_for_views(views=['B', 'R', ])
arrange_VR5Gait_for_views(views=['B', 'L', ])
arrange_VR5Gait_for_views(views=['F', 'L', ])
arrange_VR5Gait_for_views(views=['B', 'R', 'L', ])
arrange_VR5Gait_for_views(views=['F', 'B', 'L', ])

get_npys_auto.get_npys(view_list=['B', 'L', 'BR', 'BL', 'FL', 'BRL', 'FBL'], frame_list=[30, ])
