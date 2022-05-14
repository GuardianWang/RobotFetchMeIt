# RobotFetchMeIt
Brown university [CSCI 2952-O](http://cs.brown.edu/courses/csci2952o/) final project.

Given a depth map and a text description, we use [Spot](https://www.bostondynamics.com/products/spot)
to detect the object that matches the description in the scene and walk to it.

This repo is based on [imvotenet](https://github.com/facebookresearch/imvotenet).

# Group members

- [Zichuan Wang](https://github.com/GuardianWang)
- [Yiwen Chen](https://github.com/yiwenchen1999/)
- [Rao Fu](https://github.com/FreddieRao)
- [Xinyu Liu](https://github.com/jasonxyliu)

# Demo

- Demo only with 3D object detector

[![](https://img.youtube.com/vi/LdnsRqQYvwg/3.jpg)](https://youtu.be/LdnsRqQYvwg)

# Resources

[spot-sdk](https://github.com/jasonxyliu/spot-sdk)

# Installation

Clone this repository and submodules:
```bash
git clone https://github.com/GuardianWang/imvotenet.git RobotFetchMeIt
cd RobotFetchMeIt
git submodule init
```
We use conda to create a virtual environment.
Because loading the weights of text model requires pytorch>=1.8 and compiling pointnet2 requires pytorch==1.4, we are
using two conda environments.
```bash
conda create -n fetch python=3.7 -y
conda create -n torch18 python=3.7 -y
```
Overall, the installation is similar to [VoteNet](https://github.com/facebookresearch/votenet). 
GPU is required. The code is tested with Ubuntu 18.04, Python 3.7.13, PyTorch 1.4.0, CUDA 10.0, cuDNN v7.4, and NVIDIA GeForce RTX 2080.

Before installing Pytorch, make sure the machine has GPU on it. Check by `nvidia-smi`. 
We need GPU to compile PointNet++.
First install [PyTorch](https://pytorch.org/get-started/locally/), 
for example through [Anaconda](https://docs.anaconda.com/anaconda/install/):
Find cudatoolkit and cudnn version match [here](https://developer.nvidia.com/rdp/cudnn-archive).
Find how to let nvcc detect a specific cuda version [here](https://stackoverflow.com/questions/40517083/multiple-cuda-versions-on-machine-nvcc-v-confusion).
```bash
conda activate fetch
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=9.2 cudnn=7.6.5 -c pytorch
pip install matplotlib opencv-python plyfile tqdm networkx==2.2 trimesh==2.35.39
pip install protobuf
pip install open3d

conda activate torch18
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 cudnn=7.6.5 -c pytorch
pip install ordered-set
pip install tqdm
pip install pandas
```

The code depends on [PointNet++](http://arxiv.org/abs/1706.02413) as a backbone, which needs compilation.
We need to compile by the [arch](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) of GPU we want to use.
```bash
conda activate fetch
cd pointnet2
TORCH_CUDA_ARCH_LIST="7.5" python setup.py install
cd ..
```

## Data
Please follow the steps listed [here](https://github.com/facebookresearch/votenet/blob/master/sunrgbd/README.md) to set up the SUN RGB-D dataset in the `sunrgbd` folder. 
To prevent a bug in MATLAB, we need to 
change line 10 in `OFFICIAL_SUNRGBD/SUNRGBDtoolbox/readData/read_3d_pts_general.m` to
```
rgb = double(im) / 255;
```

The expected dataset structure under `sunrgbd` is:
```
sunrgbd/
  sunrgbd_pc_bbox_votes_50k_{v1,v2}_{train,val}/
  sunrgbd_trainval/
    # raw image data and camera used by ImVoteNet
    calib/*.txt
    image/*.jpg
```
For ImVoteNet, we provide 2D detection results from a pre-trained Faster R-CNN detector [here](https://dl.fbaipublicfiles.com/imvotenet/2d_bbox/sunrgbd_2d_bbox_50k_v1.tgz). Please download the file, uncompress it, and place the resulting folders (`sunrgbd_2d_bbox_50k_v1_{train,val}`) under `sunrgbd` as well.

## Training and Evaluation

To submit a task in grid, cd into `imvotenet` and run 
```
psub runme
```
Note that the GPU we specify matches the arch when compiling PointNet++.

Once the code and data are set up, one can train ImVoteNet by the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_imvotenet --log_dir log_imvotenet
```
The setting `CUDA_VISIBLE_DEVICES=0` forces the model to be trained on a single GPU (GPU `0` in this case). With the default batch size of 8, it takes about 7G memory during training. 

To reproduce the experimental results in the paper and in general have faster development cycles, one can use a shorter learning schedule: 
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --use_imvotenet --log_dir log_140ep --max_epoch 140 --lr_decay_steps 80,120 --lr_decay_rates 0.1,0.1
```

As a baseline, this code also supports training of the original VoteNet, which is launched by:
```bash
CUDA_VISIBLE_DEVICES=2 python train.py --log_dir log_votenet
```
In fact, the code is based on the VoteNet repository at commit [2f6d6d3](https://github.com/facebookresearch/votenet/tree/2f6d6d3), as a reference, it gives around 58 mAP@0.25.

For other training options, one can use `python train.py -h` for assistance.

After the model is trained, the checkpoint can be tested and evaluated on the `val` set via:
```bash
python eval.py --use_imvotenet --checkpoint_path log_imvotenet/checkpoint.tar --dump_dir eval_imvotenet --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```
For reference, ImVoteNet gives around 63 mAP@0.25.

## TODO
- Add docs for some functions
- Investigate the 0.5 mAP@0.25 gap after moving to PyTorch 1.4.0. (Originally the code is based on PyTorch 1.0.)

## LICENSE

The code is released under the [MIT license](LICENSE).
