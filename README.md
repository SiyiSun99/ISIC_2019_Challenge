# ISIC_2019_Challenge
This repository contains code for CSCI-GA.2271-001 Computer Vision's final project. We don't upload any images to this repository. All images belongs to ISIC. You can download the [dataset here](https://link-url-here.org). You should clone this repository and form the folder as the following 

##[sub-section](./README.md#Folder Structure)    

## Abstract 
Melanoma is the deadliest form of skin cancer (5,000,000 newly diagnosed cases in the United States every year) responsible for an overwhelming majority of skin cancer deaths. In 2015, the global incidence of melanoma was estimated to be over 350,000 cases, with almost 60,000 deaths. Although the mortality is significant, when detected early, melanoma survival exceeds 95\%. Our task is to classify dermoscopic images among nine different diagnostic categories (8 for different skin diseases and 1 for normal skin). In this paper We employed EfficientNet, the best and the most efficient CNN model present currently, to address this multi-classification task. Realtime data augmentation, which uses random rotation, translation, shear, and zoom within specified bounds is used to increase the number of available training samples. Reblanced weights for cross-entry loss is adopted to better approximate actual probability distributions. The performance of our model beats the best model with accuracy 0.636 in the leaderboard of ISIC 2019 Challenge with validation accuracy of 0.658.

## Folder Structure
