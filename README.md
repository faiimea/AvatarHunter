# AvatarHunter


[AvatarHunter](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10229062) 


## Prerequisites

- Python 3.6
- PyTorch 0.4+
- GPU


## Getting started



### Dataset & Preparation
Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)

Download [TrainData_Sys4](https://drive.google.com/file/d/1zwt9GAx8qGd7Kr713ioFffqw7ivkEDmW/view?usp=sharing)

Download [TestData_Sys4](https://drive.google.com/file/d/1Yw3hrHY3VCUGC-qEVSX2Uy1ON3FzmYSh/view?usp=sharing)

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
- **Step1:** Organize the directory as: 
`your_dataset_path/subject_ids/walking_conditions/views`.
E.g. `CASIA-B/001/nm-01/000/`.
- **Step2:** Cut and align the raw silhouettes with `pretreatment.py`.
(See [pretreatment](#pretreatment) for details.)
Welcome to try different ways of pretreatment but note that
the silhouettes after pretreatment **MUST have a size of 64x64**.


#### Pretreatment
`pretreatment.py` uses the alignment method in
[this paper](https://ipsjcva.springeropen.com/articles/10.1186/s41074-018-0039-6).
Pretreatment your dataset by
```
python pretreatment.py --input_path='root_path_of_raw_dataset' --output_path='root_path_for_output'
```
- `--input_path` **(NECESSARY)** Root path of raw dataset.
- `--output_path` **(NECESSARY)** Root path for output.
- `--log_file` Log file path. #Default: './pretreatment.log'
- `--log` If set as True, all logs will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1

### Configuration 

In `config.py`, you might want to change the following settings:
- `dataset_path` **(NECESSARY)** root path of the dataset 
(for the above example, it is "gaitdata")
- `WORK_PATH` path to save/load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs

### Train
Train a model by
```bash
python train.py
```
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE

### Evaluation
Evaluate the trained model by
```bash
python test.py
```
- `--iter` iteration of the checkpoint to load. #Default: 80000
- `--batch_size` batch size of the parallel test. #Default: 1
- `--cache` if set as TRUE all the test data will be loaded at once before the transforming start.
This might accelerate the testing. #Default: FALSE

It will output Rank@1 of all three walking conditions. 
Note that the test is **parallelizable**. 
To conduct a faster evaluation, you could use `--batch_size` to change the batch size for test.


## To Do List
- Update DataSet SysGait -- 2024.10.10


## Citation
Please cite these papers in your publications if it helps your research:
```
@inproceedings{meng2023anonymization,
  title={De-anonymization attacks on metaverse},
  author={Meng, Yan and Zhan, Yuxia and Li, Jiachun and Du, Suguo and Zhu, Haojin and Shen, Xuemin Sherman},
  booktitle={IEEE INFOCOM 2023-IEEE Conference on Computer Communications},
  pages={1--10},
  year={2023},
  organization={IEEE}
}
```



