This code is a re-implementation of the following paper:
Mengran Gou, Fei Xiong, Octavia Camps, Mario Sznaier. "MoNet: Moment embedding network", CVPR 2018

The code is tested on Ubuntu 14.04 using a single NVIDIA Titan X GPU, 64G RAM and MATLAB R2016b. 

# Dependency
- VLFEAT
- MatConvNet
- [VGG-VG16 model](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat)

# How to run
- Download the pretrained VGG model and put it under ./data/models
- Download one dataset from the following list and put it under ./data
- Modify the paths in './base/setup.m'
- Run "script_run_exp.m". The default setting will train/test MoNet on CUB

# Supported datasets
[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
[FGVC aircraft](http://www.robots.ox.ac.uk/~vgg/data/oid/)
[Stanford cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

# Known issues
- If you kill the code during the pre-trained feature extraction stage, you have to delet ./data/%NETWORK%/%DATASET%/nonftGlbGau and then re-start

# Acknowledgement
This code is modified from the G2DeNet package and based on BCNN code. Please refer to the originial websites for more details
[G2DeNet](http://peihuali.org/publications/G2DeNet/G2DeNet-FGVC-v1.0.zip)
[BCNN](https://bitbucket.org/tsungyu/bcnn)
