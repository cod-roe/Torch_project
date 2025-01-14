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
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from dataclasses import dataclass
from contextlib import contextmanager
# %%


@dataclass
class ModelConfig:
    model_name: str
    model_fn: callable  # efficientnet_v2_s
    final_in_features: int  # =1280 1408 1280
    num_class: int = 2
    input_channels: int = 6


class EfficientNetV2(nn.Module):
    def __init__(self, config: ModelConfig):  #
        super().__init__()
        # EfficientNetV2-Sの事前学習済みモデルをロード
        base_model = config.model_fn(pretrained=True)

        # 最初の畳み込み層を取得し、6層にカスタマイズ
        original_conv = base_model.features[0][0]  # 最初のConv2d
        base_model.features[0][0] = nn.Conv2d(
            in_channels=config.input_channels,  # 入力チャンネル数を6に変更
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # エンコーダ部分
        self.encoder = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # サイズ調整
            nn.Flatten(1),
        )

        # 分類層
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.final_in_features, config.num_class),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

#%%
# %%
# model

# dataclass設定
efficientnet_v2_s_config = ModelConfig(
    model_name="EfficientNetV2-S",
    model_fn=efficientnet_v2_s,
    final_in_features=1280,
    num_class=2,  # 出力の数
    input_channels=6,  # カスタム入力チャネル数
)
# EfficientNetV2-Sのモデル
model = EfficientNetV2(efficientnet_v2_s_config)


# デバイスを選択
device = (
    "cuda"
    if torch.cuda.is_available()
    else xm.xla_device()
    if "TPU" in str(xm.xla_device())
    else "cpu"
)
print(device)
# デバイスに移動
model = model.to(device)


# weights =[23.8050,  1.0438]
print(f"weight:{weights}")
print(f"eta_min:{eta_min}")

# learning settings
criterion = nn.CrossEntropyLoss(weights)
optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)  #



# 自動混合精度のためのコンテキストマネージャ
@contextmanager
def dummy_autocast():
    yield

if device == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    autocast = torch.autocast
else:
    scaler = None
    autocast = dummy_autocast
    
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)  # 学習率の変化


# %%
# 学習の関数定義
# confusion_matrix版
def train_model(
    model,
    start_epoch,
    stop_epoch,
    epochs,
    dataloaders_dict,
    criterion,
    optimizer,
    freeze_backbone=False,
):
    """
    検証時のベストスコアを更新したときに、そのエポック時点のモデルパラメータを保存するようにコーディング。
    2025/1/14
    L2を実装
    """
    best_iou = 0.0
    loss_dict = {"train": [], "val": []}
    iou_dict = {"train": [], "val": []}
    if freeze_backbone:
        # 【初期学習】バックボーンのパラメータを固定
        for param in model.encoder[0].parameters():
            param.requires_grad = False
    elif not freeze_backbone:
        # 【追加学習】バックボーンのパラメータを解凍
        for param in model.encoder[0].parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, stop_epoch):
        logger.info(
            "=" * 20 + f"Epoch: {epoch+1} / {stop_epoch}(全{epochs})Start!" + "=" * 20
        )
        logger.info(f"現在の学習率: {optimizer.param_groups[0]['lr']}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                # optimizer.train()
            else:
                model.eval()
                # optimizer.eval()
            epoch_loss = 0.0
            pred_list = []
            true_list = []
            for images, labels in tqdm(dataloaders_dict[phase]):
                images = images.float().to(
                    device, non_blocking=True
                )  # データを非同期的に GPU に転送
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()  # 勾配のリセット
                with torch.set_grad_enabled(phase == "train"):
                    with autocast(device_type=device):  # autocastのコンテキスト
                        # 出力
                        outputs = model(images)
                        # 損失計算
                        loss = criterion(outputs, labels)
                        # L2正則化
                        # l2norm = weight_decay * sum(torch.sum(w**2) for w in model.parameters() if w.requires_grad and w.ndim > 1)
                        # loss = loss + l2norm
                        _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        if scaler:
                            # 逆伝播と勾配更新（GradScalerを使用）
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            xm.optimizer_step(optimizer)


                    epoch_loss += loss.item() * images.size(0)
                    preds = preds.to("cpu").numpy()
                    pred_list.extend(preds)
                    labels = labels.to("cpu").numpy()
                    true_list.extend(labels)

            if phase == "train":
                scheduler.step()  # エポック終了時に学習率更新

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            tn, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
            epoch_iou = tp / (tp + fp + fn)
            loss_dict[phase].append(epoch_loss)
            iou_dict[phase].append(epoch_iou)
            logger.info(f"{phase}_Loss: {epoch_loss:.5f} {phase}_IoU: {epoch_iou:.4f}")

            # パラメータの保存
            if (phase == "val") and (
                (epoch_iou > best_iou)
                or ((epoch + 1) == epochs)
                or ((epoch + 1) == stop_epoch)
            ):
                best_iou = epoch_iou
                checkpoint_path = (
                    EXP_MODEL / f"{name}_epoch{epoch+1}_iou_{epoch_iou:.4f}.pth"
                )

                # optimizer.eval()#なくてもいい
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "epoch_iou": epoch_iou,
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"ベストを更新しました（or last epoch） {name}_epoch{epoch+1}_iou_{epoch_iou:.4f}.pth"
                )

    print("well Done!!")
    return loss_dict, iou_dict




# %%

# 保存したパラメータの読み込み
last_epoch = 10  # 最後のエポック数回数
epoch_iou = 0.6816
checkpoint_path = EXP_MODEL / f"{name}_epoch{last_epoch}_iou_{epoch_iou:.4f}.pth"
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]

# CosineAnnealingLR の再設定
scheduler = CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=eta_min, last_epoch=start_epoch - 1
)


# start_epoch = 5
print(f"Resuming from epoch {start_epoch} ")

# %%
# モデルの学習
stop_epoch = 15


# TPUの8コアを使ってプロセスを立ち上げる
FLAGS = {
    'num_epochs': 30,  # 全体のエポック数
    'start_epoch': 10,  # 開始エポック
    'stop_epoch': 20   # 終了エポック
}
xmp.spawn(train_model, args=(FLAGS,), nprocs=8)


@dataclass
class ModelConfig:
    model_name: str
    model_fn: callable  # convnext_base convnext_large
    final_in_features: int  # =1024 1536
    num_class: int = 2
    input_channels: int = 6



new_loss_dict, new_iou_dict = train_model(
    model=model,
    start_epoch=start_epoch,
    stop_epoch=stop_epoch,
    epochs=epochs,
    dataloaders_dict=dataloaders_dict,
    criterion=criterion,
    optimizer=optimizer,
    freeze_backbone=False,
)


#%%
num_workers=os.cpu_count()