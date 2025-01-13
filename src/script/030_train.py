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

#%% Utilities
# Utilities 
# =================================================

#seedの固定
# =================================================

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random_state = seed
    random.seed(seed)                     # Python標準のランダムシード
    np.random.seed(seed)                  # NumPyのランダムシード
    torch.manual_seed(seed)               # PyTorchのランダムシード（CPU用）
    torch.cuda.manual_seed(seed)          # PyTorchのランダムシード（GPU用）
    torch.cuda.manual_seed_all(seed)      # PyTorchのランダムシード（マルチGPU用）
    torch.backends.cudnn.deterministic = True  # 再現性のための設定
    torch.backends.cudnn.benchmark = False     # 再現性のための設定



# 今の日時
# =================================================
def dt_now():
    jst = dt.timezone(dt.timedelta(hours=9))  # 日本標準時 (UTC+9)
    dt_now = dt.datetime.now(jst)  # 日本時間を指定
    return dt_now

#ログ保存　 stdout と stderr をリダイレクトするクラス
# =================================================
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# make dirs
# =================================================
def make_dirs():
    for d in [EXP_MODEL]:
        os.makedirs(d, exist_ok=True)
    print("フォルダ作成完了")

# load img
# =================================================
def load_img(path):
    img_bgr = cv2.imread(path)
    img_rgb = img_bgr[:, :, ::-1]
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb


# Pickle形式で保存
# =================================================

def save_metrics_pickle(loss_dict, iou_dict, filename="metrics.pkl"):
    data = {"loss": loss_dict, "iou": iou_dict}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Metrics saved to {filename}")

# Pickle形式で読み込み
# =================================================
def load_metrics_pickle(filename=save_path):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Metrics loaded from {filename}")
    return data["loss"], data["iou"]

# データのアップデート
def update_metrics(loss_dict, iou_dict, new_loss, new_iou):
    loss_dict['train'].extend(new_loss['train'])
    loss_dict['val'].extend(new_loss['val'])
    iou_dict['train'].extend(new_iou['train'])
    iou_dict['val'].extend(new_iou['val'])



# 可視化関数 (Seaborn使用) lossとmetrics
# =================================================
def plot_metrics_sns(loss_dict, iou_dict,name=name,  output_dir=OUTPUT_EXP):
    epochs = range(1, len(loss_dict["train"]) + 1)

    # DataFrameに変換
    loss_data = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': loss_dict["train"],
        'Validation Loss': loss_dict["val"]
    })

    iou_data = pd.DataFrame({
        'Epoch': epochs,
        'Train IoU': iou_dict["train"],
        'Validation IoU': iou_dict["val"]
    })

    # Lossのプロット
    plt.figure(figsize=(12, 5))

    # Train/Validation Lossのプロット
    plt.subplot(1, 2, 1)
    sns.lineplot(data=loss_data, x='Epoch', y='Train Loss', label="Train Loss", marker='o')
    sns.lineplot(data=loss_data, x='Epoch', y='Validation Loss', label="Validation Loss", marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # y軸の範囲を0から0.015に設定
    plt.ylim(0.002, 0.01)

    # IoUのプロット
    plt.subplot(1, 2, 2)
    sns.lineplot(data=iou_data, x='Epoch', y='Train IoU', label="Train IoU", marker='o')
    sns.lineplot(data=iou_data, x='Epoch', y='Validation IoU', label="Validation IoU", marker='o')
    plt.title("IoU Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")

    plt.legend()

    plt.tight_layout()

    # 画像の保存
    plt.savefig(f"{output_dir}/{name}_transition.png", format="png", dpi=300)
    plt.show()

    return plt




# 可視化関数 ヒートマップとROC
# =================================================
def plot_heatmap_roc(cm,true_list, pred_list,name=name,  output_dir=OUTPUT_EXP):
    """true_list, pred_list
    ROC曲線をプロットし、画像として保存する。

    Args:
        similarities (list): 類似度リスト。
        labels (list): ラベルリスト。
        output_path (str): プロット画像の保存パス。
    """
    # ヒートマップの描画
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred_0", "Pred_1"], yticklabels=["True_0", "True_1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    #ROC Curveの描画
    fpr, tpr, thresholds = roc_curve(true_list, pred_list)
    auc = roc_auc_score(true_list, pred_list)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    # plt.savefig(OUTPUT_EXP + f"/{name}_roc-curve.png", format="png", dpi=300)  # 画像を保存
    plt.savefig(f"{output_dir}/{name}_heatmap_roc.png", format="png", dpi=300)
    plt.show()


# 可視化関数 precision_recall_curveとAP
# =================================================
def plot_pr(true_list, pred_list):

    # PR曲線の計算
    precision, recall, thresholds = precision_recall_curve(true_list, pred_list)

    # APスコアの計算
    ap_score = average_precision_score(true_list, pred_list)

    pr_data = pd.DataFrame({
        'Precision': precision[:-1],  # 最後の点は使用しない
        'Recall': recall[:-1],
        'Threshold': thresholds
    })

    # Seabornでプロット
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    sns.lineplot(data=pr_data, x='Recall', y='Precision', label=f"AP Score = {ap_score:.2f}")
    plt.title("Precision-Recall Curve", fontsize=16)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

    print(f"APスコア: {ap_score}")

    # 閾値の選定例: F1スコアが最大となる閾値を選択
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # F1スコア計算
    best_threshold_idx = f1_scores.argmax()
    best_threshold = thresholds[best_threshold_idx]

    print(f"最適な閾値: {best_threshold}")




# %%
"""学習の関数定義 2025/01/11"""
# confusion_matrix版
def train_model(model, start_epoch, stop_epoch,epochs, dataloaders_dict, criterion, optimizer, freeze_backbone=False):
    #  検証時のベストスコアを更新したときに、そのエポック時点のモデルパラメータを保存するようにコーディング。
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

        logger.info("="*20 + f"Epoch: {epoch+1} / {stop_epoch}(全{epochs})Start!"+"="*20)
        logger.info(f"現在の学習率: {optimizer.param_groups[0]['lr']}" )

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
                images = images.float().to(device, non_blocking=True) # データを非同期的に GPU に転送
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad() # 勾配のリセット
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type=device):  # autocastのコンテキスト
                        # 出力
                        outputs = model(images)
                        # 損失計算
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        #逆伝播と勾配更新（GradScalerを使用）
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

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


            #パラメータの保存
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
                print(f"Model checkpoint saved at {checkpoint_path}")

    print("well Done!!")
    return loss_dict, iou_dict


# %%
# Pickle形式で保存
def save_metrics_pickle(loss_dict, iou_dict, filename="metrics.pkl"):
    data = {"loss": loss_dict, "iou": iou_dict}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Metrics saved to {filename}")


# %%
# lossとmatricsの保存

#呼び出し
loss_dict, iou_dict = load_metrics_pickle(save_path)

#アップデート
update_metrics(loss_dict, iou_dict, new_loss_dict, new_iou_dict)

print(f"loss:{len(loss_dict)}")
print(loss_dict)
print(f"Iou:{len(iou_dict)}")
print(iou_dict)

#保存
save_metrics_pickle(loss_dict, iou_dict, save_path)
# %%
#可視化 Loss Iou
plot_metrics_sns(loss_dict, iou_dict)

# 終了時刻
print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

# %%
plot_metrics(loss_dict, iou_dict)

loss_dict = {"train": [1.0877887378625233, 0.8020419957363792, 0.7120,0.6586,0.6152, 0.6038,0.6034,0.5751,0.5760,0.5557,0.5480,0.5487,0.5544,0.5394,0.5277,0.5239,0.5096,0.5047,0.5000,0.5029], "val": [0.8311493527667214, 0.6297086958114837,0.4062,0.5278,0.3473,0.4535,0.4705,0.5044,0.4129, 0.4060,0.4655,0.4042,0.3991,0.3044,0.3097,0.4709,0.3730,0.5781,0.3726,0.2938]}
iou_dict = {"train": [0.36716777857915617, 0.43990121857414677,0.4823 , 0.5114,0.5287,0.5392,0.5428,0.5529,0.5551,0.5679,0.5692,0.5666,0.56420,0.5724,0.5789,0.5833,0.5843,0.5938,0.5985,0.5968], "val": [0.10538832883714326, 0.12023055690858905,0.2198,0.1715,0.2534,0.1788,0.1728,0.3946,0.2049,0.2098,0.2030,0.2845,0.2428,0.2952,0.2865,0.3560,0.3963,0.1311,0.2590,0.2976]}