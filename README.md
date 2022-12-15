# ISIC_2019_Challenge
This repository contains code for CSCI-GA.2271-001 Computer Vision's final project. We don't upload any images to this repository. All images belong to ISIC. You can download the [dataset here](https://link-url-here.org). You should clone this repository and form the folder as the following ["Folder Struture"](https://github.com/SiyiSun99/ISIC_2019_Challenge/blob/main/README.md#folder-structure) section.  

## Abstract 
Melanoma is the deadliest form of skin cancer (5,000,000 newly diagnosed cases in the United States every year) responsible for an overwhelming majority of skin cancer deaths. In 2015, the global incidence of melanoma was estimated to be over 350,000 cases, with almost 60,000 deaths. Although the mortality is significant, when detected early, melanoma survival exceeds 95\%. Our task is to classify dermoscopic images among nine different diagnostic categories (8 for different skin diseases and 1 for normal skin). In this paper We employed EfficientNet, the best and the most efficient CNN model present currently, to address this multi-classification task. Realtime data augmentation, which uses random rotation, translation, shear, and zoom within specified bounds is used to increase the number of available training samples. [Focal Loss function](https://arxiv.org/abs/1708.02002) is adopted to addresses class imbalance during training in this multi-classification task, which applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified skin disease types. With the help of [Ray tune](https://docs.ray.io/en/latest/tune/index.html), the performance of our model beats the best model with accuracy 0.634 in the [leaderboard of ISIC 2019 Challenge](https://challenge.isic-archive.com/leaderboards/2019/) with test accuracy of 0.646 after the hyper-parameter tuning.

## Graphic Abstract
![image](https://user-images.githubusercontent.com/98569478/207809768-74677bae-e2f3-462a-9183-24d4c7487832.png)

## Hyper-parameter Tuning
![tune_high](https://user-images.githubusercontent.com/98569478/207810080-7e5bd70f-6e65-42ea-97ff-20a6ca333be2.jpg)


## Folder Structure
```
Structure of the folder
|   efficient.py
|   requirements.txt
|   ISIC_2019_Test_Metadata.csv
|   ISIC_2019_Training_GroundTruth.csv
|   ISIC_2019_Training_Metadata.csv
|   
\---Dataset_processed
    +---ISIC_2019_Test_Input
    |       ISIC_0034321.jpg
    |       ISIC_0034322.jpg
    |       ...
    |       ISIC_0073253.jpg
    |       (8,239 images in total)
    |       
    \---ISIC_2019_Training_Input
            ISIC_0000000.jpg
            ISIC_0000001.jpg
            ...
            ISIC_0073254.jpg
            (25,331 images in total)
```

## Requirements
```
albumentations==1.0.3
matplotlib==3.5.3
numpy==1.21.6
pandas==1.3.5
Pillow==9.3.0
ray==2.1.0
scikit_learn==1.2.0
seaborn==0.12.1
torch==1.12.0
tqdm==4.64.1
```

