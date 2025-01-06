# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

import gc
import glob
from IPython.display import display
import logging

# import re
import os
import random
import sys
from pathlib import Path

# import pickle
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
import japanize_matplotlib

import cv2
from PIL import Image
from skimage import io


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from torch.optim.lr_scheduler import CosineAnnealingLR


from sklearn.model_selection import train_test_split  # StratifiedKFold , KFold
from sklearn.metrics import (
    confusion_matrix,
)  # ,mean_squared_error,accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# %%
######################
# set dirs & filename
######################
comp_name = "Satellite"

if "google.colab" in sys.modules:  # colab環境
    print("google.colab")
    INPUT_PATH = Path("/content")  # 読み込みファイル場所
    # name_notebook = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'] # ノートブック名を取得
    name_notebook = "exp_EfficientNetV2-S_1225.ipynb"
    DRIVE = (
        f"/content/drive/MyDrive/Python/SIGNATE/{comp_name}"  # このファイルの親(scr)
    )


elif "kaggle_web_client" in sys.modules:  # kaggle環境
    INPUT_PATH = Path("../input/")

elif "VSCODE_CWD" in os.environ:  # vscode（ローカル）用
    print("vscode")
    INPUT_PATH = Path(f"../input/{comp_name}")  # 読み込みファイル場所
    abs_path = os.path.abspath(__file__)  # /tmp/work/src/exp/_.py'
    name_notebook = os.path.basename(abs_path)  # ノート名を取得
    DRIVE = os.path.dirname(os.getcwd())  # このファイルの親(scr)

# 共通
name = os.path.splitext(name_notebook)[0]  # 拡張子を除去　filename
OUTPUT = os.path.join(DRIVE, "output")
OUTPUT_EXP = os.path.join(OUTPUT, name)  # logなど情報保存場所
EXP_MODEL = Path(OUTPUT_EXP, "model")  # 学習済みモデル保存


######################
# ハイパーパラメータの設定
num_workers = 2  # DataLoader CPU使用量
epochs = 25
lr = 0.001  # Adam  0.001　SGD 0.005
batch_size = 256
train_ratio = 0.75
weight_decay = 5e-4
# momentum = 0.9
save_interval = 1  # 保存する間隔（エポック単位）

# %%
# confusion_matrix版

def train_model(
    model, start_epoch, stop_epoch, epochs, dataloaders_dict, criterion, optimizer
):
    #  検証時のベストスコアを更新したときに、そのエポック時点のモデルパラメータを保存するようにコーディング。
    best_iou = 0.0
    loss_dict = {"train": [], "val": []}
    iou_dict = {"train": [], "val": []}
    for epoch in range(start_epoch, stop_epoch):
        print(f"Epoch: {epoch+1} / {stop_epoch}(全{epochs})")
        print("--------------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            epoch_loss = 0.0
            pred_list = []
            true_list = []
            for images, labels in tqdm(dataloaders_dict[phase]):
                images = images.float().to(device)
                labels = labels.to(device)

                # 勾配のリセット
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    # 出力
                    outputs = model(images)
                    # 損失計算
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        # 逆伝播
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * images.size(0)
                    preds = preds.to("cpu").numpy()
                    pred_list.extend(preds)
                    labels = labels.to("cpu").numpy()
                    true_list.extend(labels)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            _, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
            epoch_iou = tp / (tp + fp + fn)
            loss_dict[phase].append(epoch_loss)
            iou_dict[phase].append(epoch_iou)
            print(f"{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}")

            # stdout と stderr を一時的にリダイレクト
            stdout_logger = logging.getLogger("STDOUT")
            stderr_logger = logging.getLogger("STDERR")

            sys_stdout_backup = sys.stdout
            sys_stderr_backup = sys.stderr

            sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
            sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
            print("-" * 20, "result", "-" * 20)
            print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

            print(f"{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}")

            # リダイレクトを解除
            sys.stdout = sys_stdout_backup
            sys.stderr = sys_stderr_backup

            if (phase == "val") and (
                (epoch_iou > best_iou)
                or ((epoch + 1) == epochs)
                or ((epoch + 1) == stop_epoch)
            ):
                best_iou = epoch_iou
                checkpoint_path = (
                    EXP_MODEL / f"{name}_epoch{epoch+1}_iou_{epoch_iou:.4f}.pth"
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "epoch_iou": epoch_iou,
                    },
                    checkpoint_path,
                )
                print(f"Model checkpoint saved at {checkpoint_path}")

    return loss_dict, iou_dict


# %%
# Pickle形式で保存
def save_metrics_pickle(loss_dict, iou_dict, filename="metrics.pkl"):
    data = {"loss": loss_dict, "iou": iou_dict}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Metrics saved to {filename}")


# %%
# 保存と読み込みのパス
save_path = OUTPUT_EXP + f"/{name}_metrics.pkl"


save_metrics_pickle(loss_dict, iou_dict, save_path)
print(loss_dict)
print(iou_dict)

# %%


# %%

# Pickle形式で読み込み
def load_metrics_pickle(filename=save_path):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Metrics loaded from {filename}")
    return data["loss"], data["iou"]


# %%
loss_dict, iou_dict = load_metrics_pickle(save_path)
print(loss_dict)
print(iou_dict)

# %%
# 学習を分けて実行
update_metrics(loss_dict, iou_dict, new_loss, new_iou)
save_metrics_pickle(loss_dict, iou_dict, save_path)
# %%
import matplotlib.pyplot as plt


# 可視化関数
def plot_metrics(loss_dict, iou_dict):
    epochs = range(1, len(loss_dict["train"]) + 1)

    # Lossのプロット
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_dict["train"], label="Train Loss")
    plt.plot(epochs, loss_dict["val"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # IoUのプロット
    plt.subplot(1, 2, 2)
    plt.plot(epochs, iou_dict["train"], label="Train IoU")
    plt.plot(epochs, iou_dict["val"], label="Validation IoU")
    plt.title("IoU Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.tight_layout()
    plt.show()


# %%
plot_metrics(loss_dict, iou_dict)

loss_dict = {"train": [1.0877887378625233, 0.8020419957363792, 0.7120,0.6586,0.6152, 0.6038,0.6034,0.5751,0.5760,0.5557,0.5480,0.5487,0.5544,0.5394,0.5277,0.5239,0.5096,0.5047,0.5000,0.5029], "val": [0.8311493527667214, 0.6297086958114837,0.4062,0.5278,0.3473,0.4535,0.4705,0.5044,0.4129, 0.4060,0.4655,0.4042,0.3991,0.3044,0.3097,0.4709,0.3730,0.5781,0.3726,0.2938]}
iou_dict = {"train": [0.36716777857915617, 0.43990121857414677,0.4823 , 0.5114,0.5287,0.5392,0.5428,0.5529,0.5551,0.5679,0.5692,0.5666,0.56420,0.5724,0.5789,0.5833,0.5843,0.5938,0.5985,0.5968], "val": [0.10538832883714326, 0.12023055690858905,0.2198,0.1715,0.2534,0.1788,0.1728,0.3946,0.2049,0.2098,0.2030,0.2845,0.2428,0.2952,0.2865,0.3560,0.3963,0.1311,0.2590,0.2976]}