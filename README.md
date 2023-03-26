# DSCANet:Identification of Underwater Targets Using the Depth-separable Convolutional Attention Module
#1. Introduction
This repository contains the source code of DSCANet for the paper: **Identification of Underwater Targets
Using the Depth-separable Convolutional Attention Module**, by Chonghua Tang and Gang Hu

#2. Updates
##first release
2023-03-24 : first release

#3. Datasets
##prepare for dataset
To run the code, you should download [ShipsEar](http://atlanttic.uvigo.es/underwaternoise/) Dataset first.
In our experiment, the downloaded ship noise files (WAV format) were first unified as the sampling rate of 48000Hz, 
then divided into 1-second segments and generated Lofargram for each segment. 
Each Lofargram was randomly cut into pictures with 192×192 size. All Lofargrams were divided into train set,
validation set and test set in a ratio of 0.6, 0.2, and 0.2, please refer to the description in the paper.
##set the path where the dataset is saved   
Configure the dataset directory for the "dataset" section in “configs/ShipsEar.json", the key of "root" is the directory where the dataset is saved.
```json
"dataset": {
    "name": "shipsear",
    "root": "/home/liubing/workdir6/selects_by_pro_randomcrop",
    "num_workers": 4,
    "classes_num": 8
  },
```
#4. Requirements
To run the code, you need to meet the following requirements

- torchvision>=0.13.1+cu116

- torch>=1.12.1+cu116

- Pillow>=7.0.0

- numpy>=1.21.5

- runx>=0.0.11

#5. Run code
The code is written in Python, and the editor recommends using pycharm. After preparing the dataset, you can run the code.

- clone projects to yur computer

- open the projects with PyCharm

- run 'train.py'


