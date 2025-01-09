# %%
"""trainsform"""
#%%
"""colobで使用"""
# コンソールに書く セッション切れ対策
# function ClickConnect(){
#   console.log("60sごとに再接続");
#   document.querySelector("colab-connect-button").click()
# }
# setInterval(ClickConnect,1000*60);
# 
# #%%
## Googleドライブをマウント
# from google.colab import drive
# drive.mount('/content/drive/')
# #%%
# !pip install japanize-matplotlib
# !pip install signate
# !pip install schedulefree

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

#%%
######################
# set dirs & filename
######################
serial_number = 10  # スプレッドシートAの番号

comp_name = "Satellite"

if 'google.colab' in sys.modules:  # colab環境
    print("google.colab")
    INPUT_PATH = Path("/content")  # 読み込みファイル場所
    # name_notebook = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'] # ノートブック名を取得
    name_notebook = "exp008_size64_0109.ipynb"
    DRIVE = f"/content/drive/MyDrive/Python/SIGNATE/{comp_name}"  # このファイルの親(scr)


elif 'kaggle_web_client' in sys.modules:  # kaggle環境
    INPUT_PATH = Path("../input/")

elif 'VSCODE_CWD' in os.environ: # vscode（ローカル）用
    print("vscode")
    INPUT_PATH =  Path(f"../input/{comp_name}")  # 読み込みファイル場所
    abs_path = os.path.abspath(__file__)  # /tmp/work/src/exp/_.py'
    name_notebook = os.path.basename(abs_path) # ノート名を取得
    DRIVE = os.path.dirname(os.getcwd())  # このファイルの親(scr)

#共通
name = os.path.splitext(name_notebook)[0] # 拡張子を除去　filename
OUTPUT = os.path.join(DRIVE, "output")
OUTPUT_EXP = os.path.join(OUTPUT, name)  # logなど情報保存場所
EXP_MODEL = Path(OUTPUT_EXP, "model")  # 学習済みモデル保存

train_dir = INPUT_PATH / "train/" #学習ファイルのフォルダ
pred_sub_dir = INPUT_PATH / "test/" #testファイルのフォルダ

save_path = OUTPUT_EXP + f"/{name}_metrics.pkl" # 保存と読み込みのパス

"""
os.pathとPathで連結の仕方が異なるので注意（そのうちもどちらかにする）
os.path 基本「+」でつなぐ
Path 「/」でつなげられる
"""

# %%

######################
# ハイパーパラメータの設定
#model_type:efficientnet_v2_s

num_workers = 2  # DataLoader CPU使用量

train_ratio = 0.9
test_ratio = 0.1

batch_size = 256
epochs = 30


# optimizer_type = "RAdam"
lr = 5e-4 # Adam  0.001　SGD 0.005
# weight_decay = 5e-4
# momentum = 0.9

#最終学習率
# scheduler :CosineAnnealingLR
eta_min=5e-6#1e-5

# 不均衡データのために重みをつける
#損失関数に重みをつける
# weights =[total / len(pos), total / len(neg)]　# クラスごとの重みを各クラスのサンプル数に基づいて調整
weights = [23.8050,  1.0438]

# %%
# Utilities #
# =================================================
# Utilities #
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



# モデルの選択
# =================================================
# def get_model(model_type):
#     if model_type == 'EfficientNetV2-S':
#         return efficientnet_v2_s(pretrained=True)
#     elif model_type == 'ResNet50':
#         return ResNet50()

# 最適化アルゴリズムの選択
# # =================================================
# def get_optimizer(optimizer_type, lr):
#     if optimizer_type == 'RAdam':
#         return optim.RAdam(model.parameters(), lr=lr)
#     if optimizer_type == 'RAdamSF':
#         return RAdamScheduleFree(model.parameters(), lr=lr, betas=(0.9, 0.999))#1e-4
#     elif optimizer_type == 'Adam':
#         return optim.Adam(model.parameters(), lr=lr)
#     elif optimizer_type == 'SGD':
#         return optim.SGD(model.parameters(), lr=lr, momentum= momentum)

#%%
def ndvi(file_path, img=None):
    '''
    ndviを算出する

    Parameters
    ---------------------
    file_path : str
        画像が格納されているファイルのパス

    Returns
    ---------------------
    ndvi_img : numpy.ndarray
        ndvi算出後画像
    '''
    # ファイル名を指定して画像を読み込む
    if file_path:
        img = io.imread(file_path)
    r_channel = img[:, :, 3]
    ur_channel = img[:, :, 4]
    ndvi_img = (ur_channel - r_channel) / (ur_channel + r_channel)
    return ndvi_img
"""使用例
ndvi_img = ndvi(file_path=None, img=img)

print(f'最小値: {ndvi_img.min()}')
print(f'最大値: {ndvi_img.max()}')
print(f'平均値: {ndvi_img.mean()}')

ndvi_scaled = ndvi_scaled[:, :, np.newaxis]
ndvi_scaled.shape
#合体
new_img = np.concatenate([img, ndvi_scaled], axis=2)
new_img.shape

"""

# %%
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
#!%matplotlib inline

# Seabornスタイル
sns.set(style="whitegrid")

# フォルダの作成
make_dirs()

# utils
# # ログファイルの設定
# logging.basicConfig(
#     filename=OUTPUT_EXP +f"/log_{name}.txt", level=logging.INFO, format="%(message)s"
# )
# # ロガーの作成
# logger = logging.getLogger()

#seedの固定
seed_everything(123)

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

weights = torch.tensor(weights).to(device)
print(weights)

# %cd /content/drive/MyDrive/Python/SIGNATE/Satellite/output
# %cd /content
