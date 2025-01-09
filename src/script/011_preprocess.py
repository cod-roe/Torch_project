# %%
"""前処理"""

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


# %%
# 明るさ変更 明るさの変動: 天候、季節、時間帯 3チャンネルver 2025/1/10
# ==================================
class AdjustBrightness:
    # 弱め0.9 ～ 1.1、強め0.7 ～ 1.3で値を調整
    def __init__(self, brightness_factor_range=(0.8, 1.2)):
        self.brightness_factor_range = brightness_factor_range

    def __call__(self, img):
        factor = random.uniform(*self.brightness_factor_range)
        return v2.functional.adjust_brightness(img, factor)


# コントラスト変更　地形や物体の種類、光の反射
class AdjustContrast:
    # 弱め0.9 ～ 1.1、強め0.7 ～ 1.3で値を調整
    def __init__(self, contrast_factor_range=(0.8, 1.2)):
        self.contrast_factor_range = contrast_factor_range

    def __call__(self, img):
        factor = random.uniform(*self.contrast_factor_range)
        return v2.functional.adjust_contrast(img, factor)


# %%
"""v2 7チャンネル使用可能 2025/01/10"""


# ランダムに[0, 90, 180, 270]のどれかの角度で回転
class RandomSpecificRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise ValueError("Input image must be a PyTorch Tensor.")
        angle = random.choice(self.angles)  # ランダムに角度を選択
        return v2.functional.rotate(img, angle)  # テンソルに回転を適用


# %%
"""v2 RGBのみ明るさ、コントラスト変更7チャンネル使用可能 2025/01/10"""


# カスタムColorJitterクラス
class CustomColorJitter:
    def __init__(
        self, brightness=0.1, contrast=0.1
    ):  # 0.2は強すぎた 0.1はまだ試していない
        self.color_jitter = v2.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, image):
        # 最初の3チャンネル（RGB）にのみ ColorJitter を適用
        rgb_channels = image[:3]
        transformed_rgb = self.color_jitter(rgb_channels)

        # 残りの4〜7チャンネルはそのまま保持
        remaining_channels = image[3:]

        # 結合して最終的な画像を作成
        return torch.cat((transformed_rgb, remaining_channels), dim=0)


# %%
"""v2 衛星画像クリップ&正規化7チャンネル使用可能 2025/01/10"""


# カスタムクリッピング + 正規化
class ClampNormalize(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        img = torch.clamp(img, min=self.min_val, max=self.max_val)  # クリッピング
        img = (img - self.min_val) / (self.max_val - self.min_val)  # スケーリング
        return img
