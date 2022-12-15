# ISIC_2019_Challenge
This repository contains code for CSCI-GA.2271-001 Computer Vision's final project. We don't upload any images to this repository. All images belong to ISIC. You can download the [dataset here](https://link-url-here.org). You should clone this repository and form the folder as the following ["Folder Struture"](https://github.com/SiyiSun99/ISIC_2019_Challenge/blob/main/README.md#folder-structure) section.  

## Abstract 
Skin diseases are one of the most common diseases, which even include melanoma, the deadliest form of skin cancer (5,000,000 newly diagnosed cases in the United States every year). In this study, we aim to classify dermoscopic images into one of nine diagnostic categories, including eight different skin diseases and one category for normal skin. To accomplish this task, we employed EfficientNet, the most efficient CNN model currently available. Real-time data augmentation, which involves applying random rotations, translations, shears, and zooms to the training data, was used to increase the number of samples available for training. In addition, we adopted [Focal Loss function](https://arxiv.org/abs/1708.02002) to address class imbalance during training in this multi-classification task, which applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified skin disease types. Our model outperformed the best model in the [leaderboard of ISIC 2019 Challenge](https://challenge.isic-archive.com/leaderboards/2019/), achieving a test accuracy of 0.646 after fine-tuning the hyperparameters with the help of [Ray tune](https://docs.ray.io/en/latest/tune/index.html).

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

