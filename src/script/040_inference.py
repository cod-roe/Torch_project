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
# 検証フェーズ
# DataLoaderを使って検証

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

# %%
"""2025/01/11"""


# 検証関数（DataLoader使用）
def predict_eval(model, eval_loader):
    eval_loss = 0.0
    pred_list = []
    true_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(eval_loader):
            images = images.float().to(device)
            labels = labels.to(device)

            outputs = model(images)
            # 損失計算
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            eval_loss += loss.item() * images.size(0)
            preds = preds.to("cpu").numpy()
            pred_list.extend(preds)
            labels = labels.to("cpu").numpy()
            true_list.extend(labels)

    epoch_loss = eval_loss / len(eval_loader.dataset)
    cm = confusion_matrix(true_list, pred_list)  # .flatten()

    # display(cm)

    # 各要素を分解
    tn, fp, fn, tp = cm.ravel()
    # tn, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
    labels = ["TN", "FP", "FN", "TP"]

    #  ヒートマップとROCの描画
    plot_heatmap_roc(cm, true_list, pred_list, name=name, output_dir=OUTPUT_EXP)

    # 混同行列の表示
    print("Confusion Matrix:")
    plt.bar(labels, [tn, fp, fn, tp], color="skyblue")
    plt.xlabel("Confusion Matrix Elements")
    plt.ylabel("Count")
    plt.title("Confusion Matrix Elements")
    plt.show()

    eval_iou = tp / (tp + fp + fn)
    logger.info(f"eval_IoU: {eval_iou:.4f}eval_loss:{epoch_loss}")
    return epoch_loss, eval_iou, true_list, pred_list


# %%
"""閾値設定用 2025/01/11"""
# 1. モデル出力を確率に変更
probs = torch.softmax(outputs, dim=1)[:, 1]  # ポジティブクラス（ゴルフ場含む）の確率を取得
probs = probs.to("cpu").numpy()
pred_list.extend(probs)  # 確率値をリストに保存


# 閾値設定を追加
def apply_threshold(probs, threshold):
    return (probs >= threshold).astype(int)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 評価する閾値
results = []

#%%
# 3. 複数の閾値で評価
for threshold in thresholds:
    pred_classes = apply_threshold(np.array(pred_list), threshold)
    cm = confusion_matrix(true_list, pred_classes)
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn)
    results.append({
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "iou": iou
    })

# 結果を表示
for res in results:
    print(f"Threshold: {res['threshold']:.1f} | IoU: {res['iou']:.4f} | TP: {res['tp']} | FP: {res['fp']} | FN: {res['fn']} | TN: {res['tn']}")

#%%
# 4. 最適な閾値を自動探索
best_threshold = 0
best_iou = 0

for res in results:
    if res["iou"] > best_iou:
        best_iou = res["iou"]
        best_threshold = res["threshold"]

print(f"Best Threshold: {best_threshold:.1f} | Best IoU: {best_iou:.4f}")


#%%

# 検証関数 全体を結合（DataLoader使用）
def predict_eval2(model, eval_loader):
    eval_loss = 0.0
    pred_list = []
    true_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(eval_loader):
            images = images.float().to(device)
            labels = labels.to(device)

            outputs = model(images)
            # 損失計算
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[
                :, 1
            ]  # ポジティブクラス（ゴルフ場含む）の確率を取得
            probs = probs.to("cpu").numpy()
            pred_list.extend(probs)  # 確率値をリストに保存

            labels = labels.to("cpu").numpy()
            true_list.extend(labels)

            eval_loss += loss.item() * images.size(0)

    epoch_loss = eval_loss / len(eval_loader.dataset)

    # 複数の閾値で評価
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 評価する閾値
    results = []

    for threshold in thresholds:
        pred_classes = apply_threshold(np.array(pred_list), threshold)
        cm = confusion_matrix(true_list, pred_classes)
        tn, fp, fn, tp = cm.ravel()
        iou = tp / (tp + fp + fn)
        results.append(
            {"threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "iou": iou}
        )

    # 最適な閾値を自動探索
    best_threshold = 0
    best_iou = 0

    for res in results:
        if res["iou"] > best_iou:
            best_iou = res["iou"]
            best_threshold = res["threshold"]

    print(f"Best Threshold: {best_threshold:.1f} | Best IoU: {best_iou:.4f}")

    final_preds = apply_threshold(np.array(pred_list), best_threshold)
    cm = confusion_matrix(true_list, pred_list)  # .flatten()
    labels = ["TN", "FP", "FN", "TP"]

    #  ヒートマップとROCの描画
    plot_heatmap_roc(cm, true_list, pred_list, name=name, output_dir=OUTPUT_EXP)

    # 混同行列の表示
    print("Confusion Matrix:")
    plt.bar(labels, cm.ravel(), color="skyblue")
    plt.xlabel("Confusion Matrix Elements")
    plt.ylabel("Count")
    plt.title("Confusion Matrix Elements")
    plt.show()

    eval_iou = best_iou
    logger.info(f"eval_IoU: {eval_iou:.4f}eval_loss:{epoch_loss}")
    return epoch_loss, eval_iou, true_list, final_preds




# %%


# %%
# 推論フェーズ
# DataLoaderを使って推論
pred_sub_dir = INPUT_PATH / "test/"

pred_sub_data = sample_submit.reset_index(drop=True)

pred_sub_dataset = SatelliteDataset(
    dir=pred_sub_dir,
    file_list=pred_sub_data,
    min=3600,
    max=23500,
    transform=ImageTransform(),
    phase="val",
    channel="ALL",
)

pred_sub_loader = DataLoader(
    pred_sub_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

# %%
"""提出用推論2025/01/11"""
# DataLoaderを使って推論
def predict_sub(model, pred_sub_loader):
    pred_list = []

    model.eval()
    with torch.no_grad():
        for images in tqdm(pred_sub_loader):
            images = images.float().to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            preds = preds.to("cpu").numpy()
            pred_list.extend(preds)

    print("Done!")
    return pred_list



#%%
"""閾値設定用 2025/01/11"""
# DataLoaderを使って推論
def predict_sub2(model, pred_sub_loader):
    pred_list = []

    model.eval()
    with torch.no_grad():
        for images in tqdm(pred_sub_loader):
            images = images.float().to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # ポジティブクラス（ゴルフ場含む）の確率を取得

            probs = probs.to("cpu").numpy
            pred_list.extend(probs)

    print("Done!")
    return pred_list


file_name = sample_submit['file_name'].values
submit_df = pd.DataFrame(data=[[f, pred] for f, pred in zip(file_name, prediction)])
submit_df.to_csv(f'{OUTPUT_EXP}/{name}_sub25.tsv', sep='\t', header=None, index=None)