# DeepCrack
This is the DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection. We provide the dataset and the pretrained model.

Zou Q, Zhang Z, Li Q, Qi X, Wang Q and Wang S, DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection, IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1498-1512, 2019. [ [PDF] ](https://cse.sc.edu/~songwang/document/tip19a.pdf)

 -  Abstract: Cracks are typical line structures that are of interest in many computer-vision applications. In practice, many cracks, e.g., pavement cracks, show poor continuity and low contrast, which bring great challenges to image-based crack detection by using low-level features. In this paper, we propose DeepCrack-an end-to-end trainable deep convolutional neural network for automatic crack detection by learning high-level features for crack representation. In this method, multi-scale deep convolutional features learned at hierarchical convolutional stages are fused together to capture the line structures. More detailed representations are made in larger scale feature maps and more holistic representations are made in smaller scale feature maps. We build DeepCrack net on the encoder-decoder architecture of SegNet and pairwisely fuse the convolutional features generated in the encoder network and in the decoder network at the same scale. We train DeepCrack net on one crack dataset and evaluate it on three others. The experimental results demonstrate that DeepCrack achieves F -measure over 0.87 on the three challenging datasets in average and outperforms the current state-of-the-art methods.

# Network Architecture
![image](https://github.com/qinnzou/DeepCrack/blob/master/figures/network.png)
# Some Results
![image](https://github.com/qinnzou/DeepCrack/blob/master/figures/intro.png)

# DeepCrack Datasets
Four datasets are used by DeepCrack. CrackTree260 is used for training, and the other three are used for test.

## CrackTree260 dataset

<div align="center">
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/6192.jpg" height="200" width="260" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/6207.jpg" height="200" width="260" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/6264.jpg" height="200" width="260" >
</div>

<div align="center">
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/6328.jpg" height="200" width="260" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/6750.jpg" height="200" width="260" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/DSCN6428.JPG" height="200" width="260" >
</div>

+ It contains 260 road pavement images - an expansion of the dataset used in [CrackTree, PRL, 2012]. These pavement images are captured by an area-array camera under visible-light illumination. We use all 260 images for training. Data augmentation has been performed to enlarge the size of the training set. We rotate the images with 9 different angles (from 0-90 degrees at an interval of 10), flip the image in the vertical and horizontal direction at each angle, and crop 5 subimages (with 4 at the corners and 1 in the center) on each flipped image with a size of 512×512. After augmentation, we get a training set of 35,100 images in total.
## CRKWH100 dataset

<div align="center">
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1000.png" height="200" width="200" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1014.png" height="200" width="200" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1022.png" height="200" width="200" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1042.png" height="200" width="200" >
</div>

<div align="center">
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1045.png" height="200" width="200" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1065.png" height="200" width="200" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1095.png" height="200" width="200" >
<img src="https://github.com/qinnzou/DeepCrack/blob/master/figures/1096.png" height="200" width="200" >
</div>

  + It contains 100 road pavement images captured by a line-array camera under visible-light illumination. The line-array camera captures the pavement at a ground sampling distance of 1 millimeter.
## CrackLS315 dataset 
  - It contains 315 road pavement images captured under laser illumination. These images are also captured by a line-array camera, at the same ground sampling distance.
## Stone331 dataset
  - It contains 331 images of stone surface. When cutting the stone, cracks may occur on the cutting surface. These images are captured by an area-array camera under visible-light illumination. We produce a mask for the area of each stone surface in the image. Then the performance evaluation can be constrained in the stone surface.


## Download:
You can download the four datasets and pretrained model from the following link,
```
CrackTree260 & GT dataset: https://1drv.ms/f/s!AittnGm6vRKLyiQUk3ViLu8L9Wzb 

CRKWH100 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
CRKWH100 GT: https://1drv.ms/f/s!AittnGm6vRKLglyfiCw_C6BDeFsP

CrackLS315 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
CrackLS315 GT: https://1drv.ms/u/s!AittnGm6vRKLg0HrFfJNhP2Ne1L5?e=WYbPvF

Stone331 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
Stone331 GT: https://1drv.ms/f/s!AittnGm6vRKLwiL55f7f0xdpuD9_
Stone331 Mask: https://1drv.ms/u/s!AittnGm6vRKLxmFB78iKSxTzNLRV?e=9Ph5aP
```

Or you can download the datasets from 
link：https://pan.baidu.com/s/1PWiBzoJlc8qC8ffZu2Vb8w 
passcodes：zfoo

## Results:
Some results on our datasets are shown as below.
![image](https://github.com/qinnzou/deepcrack/blob/master/figures/deepcrack-compare1.png)
![image](https://github.com/qinnzou/deepcrack/blob/master/figures/deepcrack-compare2.png)
![image](https://github.com/qinnzou/deepcrack/blob/master/figures/deepcrack-compare3.png)

# Set up
## Requirements
PyTorch 1.0.2 or above 
Python 3.6  
CUDA 10.0  
We run on the Intel Core Xeon E5-2630@2.3GHz, 64GB RAM and two GeForce GTX TITAN-X GPUs.

## Pretrained Models
Pretrained models on PyTorch are available at, 
link：https://pan.baidu.com/s/1WsIwVnDgtRBpJF8ktlN84A 
passcode：27py 
You can download them and put them into "./codes/checkpoints/".


```
Please notice that, as this model was trained with Pytorch, its performance is slightly different with that of the original version built on Caffe.
```

## Training 
Before training, change the paths including "train_path"(for train_index.txt), "pretrained_path" in config.py to adapt to your environment.  
Choose the models and adjust the arguments such as class weights, batch size, learning rate in config.py.  
Then simply run:  
```
python train.py 
```

## Test
To evlauate the performance of a pre-trained model, please put the pretrained model listed above or your own models into "./codes/checkpoints/" and change "pretrained_path" in config.py at first, then change "test_path" for test_index.txt, and "save_path" for the saved results.   
Choose the right model that would be evlauated, and then simply run:  
```
python test.py
```

# Citation:
Please cite our paper if you use our codes or datasets in your own work:
```
@article{zou2018deepcrack,
  title={Deepcrack: Learning Hierarchical Convolutional Features for Crack Detection},
  author={Zou, Qin and Zhang, Zheng and Li, Qingquan and Qi, Xianbiao and Wang, Qian and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1498--1512},
  year={2019},
}
```
The CrackTree260 dataset was constructed based on the CrackTree206 dataset. If you use it, please cite
```
@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}
```
# Copy Right:
This dataset was collected for academic research. 
# Contact: 
For any problem about this dataset or codes, please contact Dr. Qin Zou (qzou@whu.edu.cn).
