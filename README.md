# deepcrack
This is the source code for DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection. We provide the dataset and the pretrained model.

Zou Q, Zhang Z, Li Q, Qi X, Wang Q and Wang S, DeepCrack: Learning Hierarchical Convolutional Features for Crack Detection, IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1498--1512, 2019.

# Network Architecture
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/network.png)
# Some Results
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/1_data.jpg)
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/2_data.jpg)
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/3_data.jpg)
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/1_pred.jpg)
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/2_pred.jpg)
![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/3_pred.jpg)

# DeepCrack Dataset
## Description:
This dataset contains 19383 image sequences for lane detection, and 39460 frames of them are labeled. These images were divided into two parts, a training dataset contains 9548 labeled images and augmented by four times, and a test dataset has 1268 labeled images. The size of images in this dataset is 128*256.
+ Training set:
   - Data augmentation:
The training set is augmented. By flipping and rotating the images in three degree, the data volume is quadruple. These augmented data are separated from the original training set, which is name as “origin”. “f” and “3d” after “-” are represent for flipping and rotation. Namely, the “origin- 3df” folder is the rotated and flipped training set.
   - Data construction:
The original training set contains continuous driving scenes images, and they are divided into images sequences by every twenty. All images are contained in “clips_all”, and there are 19096 sequences for training. Each 13th and 20th frame in a sequence are labeled, and the 38192 image and their labels are in “clips_13(_truth)” and “clips_20(_truth)”.
The original training dataset has two parts. Sequences in “0313”, “0531” and “0601” subfolders are constructed on TuSimple lane detection dataset, containing scenes in American highway. The four “weadd” folders are added images in rural road in China.
+ Test set:
   - Testset #1:
The normal testset, named Testset #1, is used for testing the overall performance of algorithms. Sequences in “0530”, “0531” and “0601” subfolders are constructed on TuSimple lane dataset. 270 sequences are contained, and each 13th and 20th image is labeled.
   - Testset #2:
The Testset #2 is used for testing the robustness of algorithms. 12 kinds of hard scenes for human eyes are contained. All frames are labeled.
## Using:
Index are contained. For detecting lanes in continuous scenes, the input size is 5 in our paper. Thus, the former images are additional information to predict lanes in the last frame, and the last frame is the labeled one.
We use different sampling strides to get 5 continuous images, as shown below. Each row in the index represents for a sequence and its label for training.![image](https://github.com/qinnzou/Robust-Lane-Detection/blob/master/LaneDetectionCode/save/result/lane3.png)

## Download:
You can download the four datasets and pretrained model from the following link,
```
CrackTree260 & GT dataset: https://1drv.ms/f/s!AittnGm6vRKLyiQUk3ViLu8L9Wzb 

CRKWH100 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
CRKWH100 GT: https://1drv.ms/f/s!AittnGm6vRKLglyfiCw_C6BDeFsP

CrackLS315 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 

Stone331 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
Stone331 GT: https://1drv.ms/f/s!AittnGm6vRKLwiL55f7f0xdpuD9_
```

# Set up
## Requirements
PyTorch 0.4.0  
Python 3.6  
CUDA 8.0  
We run on the Intel Core Xeon E5-2630@2.3GHz, 64GB RAM and two GeForce GTX TITAN-X GPUs.

## Preparation
### Data Preparation
Our dataset contains 19383 continuous driving scenes image sequences, and 39460 frames of them are labeled. The size of images is 128*256. 
The training set contains 19096 image sequences. Each 13th and 20th frame in a sequence are labeled, and the image and their labels are in “clips_13(_truth)” and “clips_20(_truth)”. All images are contained in “clips_all”.  
Sequences in “0313”, “0531” and “0601” subfolders are constructed on TuSimple lane detection dataset, containing scenes in American highway. The four “weadd” folders are added images in rural road in China.  
The testset has two parts: Testset #1 (270 sequences, each 13th and 20th image is labeled) for testing the overall performance of algorithms. The Testset #2 (12 kinds of hard scenes, all frames are labeled) for testing the robustness of algorithms.   
To input the data, we provide three index files(train_index, val_index, and test_index). Each row in the index represents for a sequence and its label, including the former 5 input images and the last ground truth (corresponding to the last frame of 5 inputs).
Our dataset can be downloaded and put into "./LaneDetectionCode/data/". If you want to use your own data, please refer to the format of our dataset and indexs.

### Pretrained Models
Pretrained models on PyTorch are available using links in the Download part, including the propoesd models(SegNet-ConvLSTM, UNet-ConvLSTM) as well as the comparable two(SegNet, UNet)  
You can download them and put them into "./LaneDetectionCode/pretrained/".

## Training
Before training, change the paths including "train_path"(for train_index.txt), "val_path"(for val_index.txt), "pretrained_path" in config.py to adapt to your environment.  
Choose the models(SegNet-ConvLSTM, UNet-ConvLSTM or SegNet, UNet) and adjust the arguments such as class weights, batch size, learning rate in config.py.  
Then simply run:  
```
python train.py
```

## Test
To evlauate the performance of a pre-trained model, please put the pretrained model listed above or your own models into "./LaneDetectionCode/pretrained/" and change "pretrained_path" in config.py at first, then change "test_path" for test_index.txt, and "save_path" for the saved results.   
Choose the right model that would be evlauated, and then simply run:  
```
python test.py
```
The quantitative evaluations of Accuracy, Precision, Recall and  F1 measure would be printed, and the result pictures will be save in "./LaneDetectionCode/save/result/".  
We have put five images sequences in the "./LaneDetectionCode/data/testset" with test_index_demo.txt on UNet-ConvLSTM for demo. You can run test.py directly to check the performance.

# Citation:
Please cite our paper if you use our codes or data in your own work:
```
@article{zou2018deepcrack,
  title={Deepcrack: Learning hierarchical convolutional features for crack detection},
  author={Zou, Qin and Zhang, Zheng and Li, Qingquan and Qi, Xianbiao and Wang, Qian and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1498--1512},
  year={2019},
}
```
# Copy Right:
This dataset was collected for academic research. 
# Contact: 
For any problem about this dataset, please contact Dr. Qin Zou (qzou@whu.edu.cn).
