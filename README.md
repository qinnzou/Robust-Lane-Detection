# Robust-Lane-Detection

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
The testset has two parts: Testset #1 (270 sequences, each 13th and 20th image is labeled) for testing the overall performance of algorithms. The Testset #2 (17 kinds of hard scenes, all frames are labeled) for testing the robustness of algorithms.   
To input the data, we provide three index files(train_index, val_index, and test_index). Each row in the index represents for a sequence and its label, including the former 5 input images and the last ground truth (corresponding to the last frame of 5 inputs).
Our dataset can be downloaded here and put into "./data/". If you want to use your own data, please refer to the format of our dataset and indexs.

### Pretrained Models
Pretrained models on PyTorch are available here, including the propoesd models(SegNet-ConvLSTM, UNet-ConvLSTM) as well as the comparable two(SegNet, UNet)  
You can download them and put them into "./pretrained/".

## Training
Before training, change the paths including "train_path"(for train_index.txt), "val_path"(for val_index.txt), "pretrained_path" in config.py to adapt to your environment.  
Choose the models(SegNet-ConvLSTM, UNet-ConvLSTM or SegNet, UNet) and adjust the arguments such as class weights, batch size, learning rate in config.py.  
Then simply run:  
```
python train.py
```

## Test
To evlauate the performance of a pre-trained model, please put the pretrained model listed above or your own models into "./pretrained/" and change "pretrained_path" in config.py at first, then change "test_path" for test_index.txt, and "save_path" for the saved results.   
Choose the right model that would be evlauated, and then simply run:  
```
python test.py
```
The quantitative evaluations of Accuracy, Precision, Recall and  F1 measure would be printed, and the result pictures will be save in "save/result/".  
We have put five images sequences in the "./data/testset" with test_index_demo.txt on UNet-ConvLSTM for demo. You can run test.py directly to check the performance.

# Authorship
This project is contributed by Qin Zou group, the School of Computer Science, Wuhan University.
