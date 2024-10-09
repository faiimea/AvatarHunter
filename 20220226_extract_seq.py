import os

dir_path = "pretreat_TrainSysGait"
cmu_seq_list = []
for cmuA in os.listdir(dir_path):
	for seqB_mhC in os.listdir(os.path.join(dir_path, cmuA)):
		cmuAN = cmuA[-2:]
		seqBN = seqB_mhC[4:6]
		if "{}_{}".format(cmuAN, seqBN) not in cmu_seq_list:
			cmu_seq_list.append("{}_{}".format(cmuAN, seqBN))
cmu_seq_list.sort()
print(cmu_seq_list)
		