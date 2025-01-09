# %%
"""trainsform"""

# %%
# ライブラリ読み込み
# =================================================
import datetime as dt
import gc
import glob
from IPython.display import display
import logging

# import re
import math
import os
import random
import sys
from pathlib import Path
import pickle
from requests import get  # colab
import shutil  # colab
from tqdm import tqdm_notebook as tqdm
import warnings
# import zipfile


import numpy as np
import pandas as pd
from scipy import ndimage

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# 画像分析
import cv2
from PIL import Image
from skimage import io
# from torchvision.io import read_image

# NN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ML
from sklearn.model_selection import train_test_split  # StratifiedKFold , KFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)  # ,mean_squared_error,accuracy_score

# from schedulefree import RAdamScheduleFree
import CustomColorJitter, RandomSpecificRotation, AdjustContrast, AdjustBrightness, ClampNormalize


# %%
# 01
class ImageTransform:
    """
    画像の前処理クラス
    学習時と検証時で挙動を変える
    外れ値があるためクリッピング→正規化

    Attributes
    --------------------
    min :
    mean : tupple
        各チャネルの平均値
    std : tupple
        各チャネルの標準偏差
    """

    def __init__(self):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),  # テンソル変換
                    transforms.RandomHorizontalFlip(),  # 水平反転(ランダム)
                    transforms.RandomVerticalFlip(),  # 垂直反転(ランダム)
                    transforms.RandomAffine([-30, 30]),  # 回転(ランダム)
                    # transforms.Normalize(mean, std) #標準化
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),  # テンソル変換
                    # transforms.Normalize(mean, std) #標準化
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


# %%
"""02ver 2025/01/10
v2ではv1と処理の順番などの推奨が変更されているので注意
最初にtensor変更、[C,H,W]に変更しておく、ToTensor非推奨"""


class ImageTransform02:
    """
    画像の前処理クラス
    学習時と検証時で挙動を変える
    外れ値があるためクリッピング→正規化を最後ではなく最初に行う

    Attributes
    --------------------
    min :
    mean : tupple
        各チャネルの平均値
    std : tupple
        各チャネルの標準偏差
    """

    def __init__(self, input_size=64):  #
        self.transforms = {
            "train": v2.Compose(
                [
                    v2.ToImage(),  # [ H, W, C] 形式 から [C, H, W] 形式とtensor化
                    # CustomColorJitter(brightness=0.1, contrast=0.1),
                    ClampNormalize(
                        min_val=5000, max_val=30000
                    ),  # クリッピングと正規化 標準5000~30000
                    v2.RandomHorizontalFlip(p=0.5),  # 50%の確率で水平フリップ
                    v2.RandomVerticalFlip(p=0.5),  # 50%の確率で垂直フリップ
                    RandomSpecificRotation(
                        angles=[0, 90, 180, 270]
                    ),  # 0度、90度、180度、270度の回転
                    # v2.Pad(padding=4),  # 4ピクセルのパディング
                    # v2.RandomCrop(size=(32, 32)),  # 32x32のサイズでランダムにクロッピング
                    # v2.RandAugment(3, 9),
                    # v2.AugMix(3, 3),
                    v2.Resize((input_size, input_size)),
                    # v2.Normalize(mean, std) #標準化
                ]
            ),
            "val": v2.Compose(
                [
                    v2.ToImage(),  # [ H, W, C] 形式 から [C, H, W] 形式とtensor化
                    ClampNormalize(min_val=5000, max_val=30000),  # クリッピングと正規化
                    v2.Resize((input_size, input_size)),
                    # v2.Normalize(mean, std) #標準化
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.transforms[phase](img)
