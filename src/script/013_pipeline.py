# %%
"""Dataset"""

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
INPUT_PATH = Path("/content")  # 読み込みファイル場所
train_dir = INPUT_PATH / "train/"

# %%
"""2025/01/10"""


# Dataset
class SatelliteDataset(Dataset):
    def __init__(
        self,
        dir,
        file_list,
        transform=None,
        phase="train",
        channel="RGB",
        give_label=True,
    ):
        """
        Attributes
        -------------------
        dir : str
            画像が保管されているパス
        file_list : dataframe
            画像のファイル名とフラグが格納されているデータフレーム
        transform : torchvision.transforms.Compose
            前処理パイプライン
        phase : str
            学習か検証かを選択
        channel : str
            使用するチャネル(初期値はRGB)
        """
        self.dir = dir
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.image_path = file_list["file_name"].to_list()
        self.image_label = file_list["flag"].to_list()
        self.channel = channel
        self.give_label = give_label

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # 画像をロード
        img_path = str(Path(self.dir) / self.image_path[idx])
        # img = read_image(self.dir / img_path)  # [C, H, W] 形式　tif対応していない
        if self.channel == "ALL":
            img = io.imread(img_path)  # [ H, W, C] 形式
        elif self.channel == "RGB":
            img = io.imread(img_path)[:, :, [3, 2, 1]]  # [ H, W, C] 形式 # BGR -> RGB
        elif self.channel == "6c":
            img = io.imread(img_path)[
                :, :, [3, 2, 1, 4, 5, 6]
            ]  # [ H, W, C] 形式 # BGR -> RGB

        # 前処理の実装
        if self.transform:
            img = self.transform(img, self.phase)
        label = self.image_label[idx]

        if self.give_label:
            return img, label
        else:
            return img
