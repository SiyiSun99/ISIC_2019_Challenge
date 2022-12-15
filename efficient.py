#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import os
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.notebook import tqdm
from glob import glob

from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# #### EfficientNet Model

# <table>
#     <tr>
#         <td>
#             <img src=https://miro.medium.com/max/1400/1*rnhgFRXetwD8PvxhZIpwIA.png width=800>
#             </td><td>
#         <img src=https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_10.45.54_PM.png width=800>
#         </td></tr>
#     </table>

# In[2]:


# activation function: Swish

class Swish(nn.Module):
    """ 
    activation function: 
    Swish allows a small number of negative weights to be propagated through, 
    while ReLU (max(0, x)) thresholds all negative weights to zero.
    """
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__(*args, **kwargs)
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
class ConvBNBlock(nn.Module):
    """ 
    basic block: zero-padded 2D convolution, followed by batch 
    normalization and Swish activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args, stride=1, groups=1, **kwargs):
        super(ConvBNBlock, self).__init__(*args, **kwargs)
        padding = self._get_padding(kernel_size, stride)
        self.block = nn.Sequential(
                nn.ZeroPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                Swish(),
            )
        
    def forward(self, x):
        return self.block(x)
        
    def _get_padding(self, kernel_size, stride):
        """ add corresponding padding """
        p = np.maximum(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]
    
    
class SqueezeExcitationBlock(nn.Module):
    """ 
    The Squeeze-and-Excitation Block is an architectural unit designed to improve 
    the representational power of a network by enabling it to perform 
    dynamic channel-wise feature recalibration
    """
    def __init__(self, in_channels, reduced_dim, *args, **kwargs):
        super(SqueezeExcitationBlock, self).__init__(*args, **kwargs)
        self.squeeze_excitation = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
                    Swish(),
                    nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
                    nn.Sigmoid()
            )
    
    def forward(self, x):
        return x * self.squeeze_excitation(x)  # which is similar to swish activation
    

class MBConvBlock(nn.Module):
    """
    Inverted Linear BottleNeck layer with Depth-Wise Separable Convolution
    implements inverted residual connection like MobileNetV2
    """
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, 
                 *args, reduction_ratio=0.4, drop_connect_rate=0.2, **kwargs):
        super(MBConvBlock, self).__init__(*args, **kwargs)
        
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_channels == out_channels) & (stride == 1)
        
        assert stride in (1, 2), "stride should be 1 or 2"
        assert kernel_size in (3, 5), "kernel_size should be 3 or 5"

        hidden_dim = in_channels * expand_ratio
        reduced_dim = np.maximum(1, int(in_channels / reduction_ratio))
        
        layers = []
        if in_channels != hidden_dim:
            layers.append(ConvBNBlock(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNBlock(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            SqueezeExcitationBlock(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            residual = x
            x = self.conv(x)
            return residual + self._drop_connections(x)
        else:
            return self.conv(x)
    
    def _drop_connections(self, x):
        """ 
        dropout probability mask, works similarily as Dropout, except we 
        disable individual weights (i.e., set them to zero), instead of nodes
        """
        if not self.training:
            return x  # identity
        keep_probability = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_probability + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_probability) * binary_tensor


# In[3]:


# helper functions

def make_divisable(value, divisor=8):
    """ transform input value into closest divisable by divisor value """
    divisable_value = np.maximum(divisor, (value + divisor // 2) // divisor * divisor)
    if divisable_value < 0.9 * value:
        divisable_value += divisor
    return divisable_value

def round_filters(filters, width):
    """ return divisable number of filters """
    if width == 1.0:
        return filters
    return int(make_divisable(filters * width))  # int floor

def round_repeats(repeats, depth):
    """ calibrate number of net blocks """
    if depth == 1.0:
        return repeats
    return int(np.ceil(depth * repeats))


# In[4]:


# final model

class EfficientNet(nn.Module):
    """ Gather all blocks (it is possible to upload pretrained weights from torch hub) """
    def __init__(self, *args, width=1.0, depth=1.0, dropout=0.2, num_classes=10, **kwargs):
        super(EfficientNet, self).__init__(*args, **kwargs)
        settings = [
           # t,  c,  n, s, k  -> expand_ratio, channels, repeats, init stride, kernel_size
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        out_channels = round_filters(32, width)
        layers = [ConvBNBlock(3, out_channels, kernel_size=3, stride=2),]
        
        in_channels = out_channels
        for expand, channel, repeat, strid, kernel in settings:
            out_channels = round_filters(channel, width)
            repeats = round_repeats(repeat, depth)
            for i in range(repeats):
                stride = strid if i == 0 else 1  # reduce spatial dims only on first step
                layers.extend([
                    MBConvBlock(in_channels, out_channels, expand_ratio=expand, kernel_size=kernel, stride=stride)
                ])
                in_channels = out_channels
                
        last_channels = round_filters(1280, width)
        layers.append(ConvBNBlock(in_channels, last_channels, kernel_size=1))
        
        self.features = nn.Sequential(*layers)  # name as in torch hub
        self.classifier = nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(last_channels, num_classes)
                )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / np.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])  # flatten by mean of spatial dims
        x = self.classifier(x)
        return x


# #### Dataset

# In[5]:


# augmentations
# train: different random flips, rotations, and color shifts
def train_transforms(image_size):
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, 
                                                    sat_shift_limit=0.2, 
                                                    val_shift_limit=0.2, 
                                                    p=0.2),      
                      A.RandomBrightnessContrast(brightness_limit=0.2, 
                                                 contrast_limit=0.2, 
                                                 p=0.5)],p=0.2),
                      A.OneOf(
                              [A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5),
                               A.RandomRotate90(p=0.5),
                               A.Transpose(p=0.5),
                              ], p=0.5),
                      A.Resize(height=image_size, width=image_size, p=1),
                      A.Cutout(num_holes=6, max_h_size=10, max_w_size=10, fill_value=0, p=0.1),
                      A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
                      ToTensorV2(p=1.0),
                      ], p=1.0)

# only resize, scale [-1, 1] and converting to tensor array[h,w,c] -> tensor[c,h,w]
def valid_transforms(image_size):
    return A.Compose([A.Resize(height=image_size, width=image_size, p=1),
                      A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
                      ToTensorV2(p=1.0),
                      ], p=1.0)

# inverse trasformations of a single image-tensor
def inverse_transforms(tensor):
    tensor = tensor 
    if tensor.size(0) == 1 and len(tensor.shape) == 4:
        tensor.squeeze_(0)
    tensor = torch.clamp(tensor * 0.5 + 0.5, min=0., max=1.)
    tensor = tensor.cpu().detach().numpy().transpose(1,2,0)

    return tensor


# In[42]:


# define dataset and dataloder

class ISICDataset(Dataset):
    def __init__(self, data, transforms, device):
        self.data = data
        self.transforms = transforms
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        row = self.data.loc[ix].squeeze()
        image = Image.open(directory + row["image"] + ".jpg")
        image = np.array(image)
        
        sample = {"image": image}
        image = self.transforms(**sample)["image"]
        
        label = torch.as_tensor(row["labels"], dtype=torch.int64)
        
        return image, label
    
    def collate_fn(self, batch):
        images, labels = list(zip(*batch))
        images, labels = [[tensor[None] for tensor in subset] for subset in (images, labels)]
        images, labels = [torch.cat(subset, dim=0).to(self.device) for subset in (images, labels)]
        return images, labels


# #### Objectives and metrics

# In[43]:


class FocalLoss(nn.Module):
    """ 
    https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """
    def __init__(self, *args, 
                 alpha: torch.Tensor = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean',
                 ignore_index: int = -100,
                 **kwargs
                 ):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 2.0
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super(FocalLoss, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            c = x.shape[1]  # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)  # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
        
        y = y.long()
        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt) (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
        
def calc_accuracy(y_pred, y_true):
    return (y_true == torch.max(y_pred, 1)[1]).float().mean()


# #### Training and evaluation functions

# In[44]:


def train_one_batch(data, model, criterion, optimizer):
    model.train()
    images, labels = data
    
    optimizer.zero_grad()
    out = model(images)
    loss = criterion(out, labels)
    
    loss.backward()
    optimizer.step()
    
    accuracy = calc_accuracy(out, labels).item()
    
    return loss.item(), accuracy

@torch.no_grad()
def validate_one_batch(data, model, criterion):
    model.eval()
    images, labels = data
    
    out = model(images)
    loss = criterion(out, labels)
    
    accuracy = calc_accuracy(out, labels)
    
    return loss.item(), accuracy.item()


# In[45]:


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path='model.pth'):
        self.path = os.path.join(os.getcwd(),path)
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model=None, **kwargs):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            checkpoint = {'state_dict': model.state_dict()}
            print("##############################################################",self.path)
            torch.save(checkpoint, self.path)
            
            print(f'Model saved to: {self.path}')
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


# #### Train

# In[46]:


def train_model(config, checkpoint_dir = None):
    
    params = {"width": width, 
          "depth": depth, 
          "dropout": dropout, 
          "num_classes": num_classes
         }
    
    EPOCHS = 100
    
    model = EfficientNet(**params)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    # criterion = FocalLoss(gamma=2.0, alpha=torch.as_tensor(weights, dtype=torch.float32).to(device))  # for binary rule of thumb: alpha=0.25
    # criterion = FocalLoss(gamma=2.0, alpha=torch.as_tensor(class_weights, dtype=torch.float32).to(device)

    # weights are prety high, maybe it is hard for model to keep track for all classes, but (despite lower overall accuracy) weighted version captures more minor classes correctly
    # criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32).to(device))
    # criterion = nn.CrossEntropyLoss()  # not weighted

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-7)
    stopper = EarlyStopping(patience=5)
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=train_ds.collate_fn)
    validset = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=valid_ds.collate_fn)
    
    for epoch in range(EPOCHS):
        train_loss, train_accs = [], []
        for step, batch in enumerate(trainset, 1):
            time_1 = time.time()

            loss, accuracy = train_one_batch(batch, model, criterion, optimizer)

            train_loss.append(loss)
            train_accs.append(accuracy)

            del batch
            gc.collect()

        valid_loss, valid_accs = [], []

        for step, batch in enumerate(validset):
            loss, accuracy = validate_one_batch(batch, model, criterion)

            valid_loss.append(loss)
            valid_accs.append(accuracy)

            del batch
            gc.collect()

        print('epoch:', epoch, '/', EPOCHS,
              '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
              '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)),
              '\ttrain accuracy', '{:.4f}'.format(np.mean(train_accs)),
              '\tvalid accuracy', '{:.4f}'.format(np.mean(valid_accs)))
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        stopper(np.mean(valid_loss), model, **params)
        scheduler.step(np.mean(valid_loss))
        
        tune.report(loss = np.mean(valid_loss), accuracy = np.mean(valid_accs))
    print("Finished Training")


# In[47]:


def test_model(config, model):
    testset = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=test_ds.collate_fn)
    
    test_loss, test_accs = [], []
    for step, batch in enumerate(testset):
        loss, accuracy = validate_one_batch(batch, model, criterion)

        test_loss.append(loss)
        test_accs.append(accuracy)
        del batch
        gc.collect()

    return np.mean(test_accs)


# In[48]:


def preprocessing():
    
    data = pd.read_csv("ISIC_2019_Training_GroundTruth.csv").drop("UNK", axis=1)
    data["labels"] = data.iloc[:, 1:].idxmax(axis=1)
    data = data[~data["image"].str.contains("downsampled")]

    classes_to_int = {v: i for i, v in enumerate(data.columns[1:-1])}
    int_to_classes = {i: v for i, v in enumerate(data.columns[1:-1])}

    data["labels"] = data["labels"].map(classes_to_int)

    num_classes = len(classes_to_int)
    
    x_train, x_test = train_test_split(data, test_size=0.8, stratify=data["labels"])
    x_valid, x_test = train_test_split(x_test, test_size=0.2, stratify=x_test["labels"])

    x_train.reset_index(drop=True, inplace=True)
    x_valid.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    
    # assign higher weight for minority classes in cross-entropy loss: loss gets higher when model make mistakes on minor class
    class_weights = compute_class_weight("balanced", classes=np.unique(data["labels"]), y=data["labels"])
    # class_weights = class_weights / class_weights.sum()  # weights normalization, unneccessary
    
    return class_weights, x_train, x_valid, x_test, num_classes


# In[49]:


def run_main(max_num_epochs=400, gpus_per_trial=1, model_efficientnet = "efficientnet_b0"):
    ray.init()

    config = {
#         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.choice([0.003]),
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples = 1,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(best_trial)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    
    params = {"width": width, 
          "depth": depth, 
          "dropout": dropout, 
          "num_classes": num_classes
         }
    best_trained_model = EfficientNet(**params)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    
    best_checkpoint_dir = result.get_best_checkpoint(best_trial, "loss", "min").to_directory()
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_model(best_trial.config, best_trained_model)
    print("Best trial test set accuracy: {}".format(test_acc))


# In[41]:


# define some global parameters here
## dataset dir
directory = os.path.join(os.getcwd(),"Dataset_processed/ISIC_2019_Training_Input/")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

params = {
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}

# efficientnet_b0 params
width, depth, img_size, dropout = params["efficientnet_b1"]

class_weights, x_train, x_valid, x_test, num_classes = preprocessing()

train_ds = ISICDataset(x_train, train_transforms(img_size), device)
valid_ds = ISICDataset(x_valid, valid_transforms(img_size), device)
test_ds = ISICDataset(x_test, valid_transforms(img_size), device)

criterion = FocalLoss(gamma=2.0, alpha=None)  # not weighted

run_main()