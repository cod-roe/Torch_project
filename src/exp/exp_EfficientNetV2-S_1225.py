# %% [markdown]
## モデルの変更！
# =================================================
# EfficientNetV2-S
# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

# import gc
import glob
from IPython.display import display
import logging

# import re
import os
import random
import sys
from pathlib import Path
import pickle
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
from torch.utils.data import DataLoader, Dataset
# from torch.optim.lr_scheduler import CosineAnnealingLR


from sklearn.model_selection import train_test_split  # StratifiedKFold , KFold
# from sklearn.metrics import mean_squared_error,accuracy_score, roc_auc_score ,confusion_matrix


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 1  # スプレッドシートAの番号

######################
# Data #
######################
skip_run = False  # 飛ばす->True，飛ばさない->False

######################
# set dirs & filename
######################
comp_name = "Satellite"
# 評価：IOU 回帰 分類


if "google.colab" in sys.modules:  # colab環境
    print("google.colab")
    INPUT_PATH = Path("/content")  # 読み込みファイル場所
    # name_notebook = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'] # ノートブック名を取得
    name_notebook = "base01_Resnet18.ipynb"
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
# Dataset #
######################

######################
# ハイパーパラメータの設定
######################
# ハイパーパラメータの設定
num_epoch = 25
lr = 0.001 # Adam  0.001　SGD 0.005
batch_size = 1024
train_ratio = 0.75
weight_decay = 5e-4
momentum = 0.9
save_interval = 1  # 保存する間隔（エポック単位）

# %%
# Utilities #
# =================================================


# 今の日時
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now


# stdout と stderr をリダイレクトするクラス
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


# ファイルの確認
# =================================================
def file_list(input_path):
    file_list = []
    for dirname, _, _filenames in os.walk(input_path):
        for i, _datafilename in enumerate(_filenames):
            print("=" * 20)
            print(i, _datafilename)
            file_list.append([_datafilename, os.path.join(dirname, _datafilename)])
    file_list = pd.DataFrame(file_list, columns=["ファイル名", "ファイルパス"])
    display(file_list)
    return file_list


# %% [markdown]
## Main 分析start!
# ==========================================================
# %%
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

#!%matplotlib inline


# フォルダの作成
make_dirs()
# ファイルの確認
file_list = file_list(INPUT_PATH)

# utils
# ログファイルの設定
logging.basicConfig(
    filename=f"{OUTPUT_EXP}/log_{name}.txt", level=logging.INFO, format="%(message)s"
)
# ロガーの作成
logger = logging.getLogger()

# %%
display(torch.cuda.is_available())
torch.cuda.device_count()

# %%
# マスターデータ読み込み
# =================================================
train_master = pd.read_csv(INPUT_PATH / "train_master.tsv", sep="\t")

sample_submit = pd.read_csv(INPUT_PATH / "sample_submit.tsv", sep="\t", header=None)

# %%
train_size = len(train_master)  # 学習データのサイズ
idx = random.randint(0, train_size - 1)  # 0から学習データ数の範囲でランダムで整数を抽出
file = train_master["file_name"][idx]  # 画像ファイル名
label = train_master["flag"][idx]  # 画像ラベル

img_path = f"../input/Satellite/train/{file}"  # 画像が格納されているパス

image = io.imread(img_path)
img = io.imread(img_path)  # 画像を開く


# チャネルごとに表示
channel_label = ["1", "B", "G", "R", "5", "6", "7"]
print(f"ラベル：{label}")
fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(15, 5))
for i in range(7):
    ax[i].imshow(img[:, :, i])
    ax[i].set_title(channel_label[i])
    ax[i].set_axis_off()
plt.show()
# %%
# 前処理
"""
・正規化
・水平フリップ（水平方向に画像反転を行う）
・垂直フリップ（垂直方向に画像反転を行う）
・回転（90度、180度、270度）
"""


class Normalize:
    def __call__(self, image):
        max = 30000
        min = 5000  # 画像のピクセル値の分布を見て決める
        image_normalized = np.clip(image, min, max)
        image_normalized = (image_normalized - min) / (max - min)
        return image_normalized


class HorizontalFlip:
    def __call__(self, image):
        p = random.random()
        if p < 0.5:
            image_transformed = np.fliplr(image).copy()
            return image_transformed
        else:
            return image


class VerticalFlip:
    def __call__(self, image):
        p = random.random()
        if p < 0.5:
            image_transformed = np.flipud(image).copy()
            return image_transformed
        else:
            return image


class Rotate:
    def __call__(self, image):
        p = random.random()
        if p < 0.25:
            return image
        elif p < 0.5:
            image_transformed = ndimage.rotate(image, 90)
            return image_transformed
        elif p < 0.75:
            image_transformed = ndimage.rotate(image, 180)
            return image_transformed
        else:
            image_transformed = ndimage.rotate(image, 270)
            return image_transformed


# %%
# Data Augmentationの処理は学習時にのみに適用、transforms.Compose()と組み合わせる
# %%
class ImageTransform:
    def __init__(self):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    Normalize(),
                    HorizontalFlip(),
                    VerticalFlip(),
                    Rotate(),
                    transforms.ToTensor(),
                ]
            ),
            "val": transforms.Compose(
                [
                    Normalize(),
                    transforms.ToTensor(),
                ]
            ),
        }

    def __call__(self, image, phase="train"):
        return self.data_transform[phase](image)


# 前処理を設計
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # サイズ変更
    transforms.ToTensor(),          # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

# 例：データセットの前処理で使用
dataset = CustomDataset(transform=transform)


# %%

"""
1. __init__:初期化を行う。
2. __len__:1エポックあたりに使用するデータ数を返す。
3. __getitem__:データの読み込み、前処理を行った上で、入力画像と正解ラベルのセットを返す。

"""


# %%
# データの順番は？




# DataLoaderの作成
# =================================================
# DataLoader
trainloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
validloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# %%
# モデルの定義
# =================================================

# import timm
# # PreResNet-18をロード
# model = timm.create_model('resnet18', pretrained=True)
# # SE-ResNet-50のロード
# model = timm.create_model('seresnet50', pretrained=True)
# EfficientNet-B0をロード
# model = models.efficientnet_b0(pretrained=True)

# EfficientNetV2-Sをロード
# model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
model = models.efficientnet_v2_s(pretrained=True)

# %%
# タスクに合わせてレイヤを調整
# =================================================

# モデルの入力層の再定義 入力チャンネル数を7に
original_conv = model.features[0][0]
model.features[0][0] = nn.Conv2d(
    in_channels=7,  # 入力チャンネル数を7に変更
    out_channels=original_conv.out_channels,  # 元の出力チャンネル数をそのまま使用
    kernel_size=original_conv.kernel_size,  # 元のカーネルサイズをそのまま使用
    stride=original_conv.stride,  # 元のストライドをそのまま使用
    padding=original_conv.padding,  # 元のパディングをそのまま使用
    bias=original_conv.bias is not None,  # 元のバイアスの設定をそのまま使用
)
# 7:チャンネル数　24：初期フィルタ数
# モデルの出力層の再定義
model.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# デバイスに移動
model = model.to(device)

# %%
model

# %%
# 最適化アルゴリズムと損失関数の設定
optimizer = optim.Adam(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()

#%%
# 評価指標IoUの設計(pytorchのtensorを想定)
def IoU(predicts, labels):
    outs = predicts.max(1)[1]
    # TP
    p_true_ids = torch.nonzero(outs)  # 正例と予測したインスタンスのインデックス番号
    tp = (outs[p_true_ids] == labels[p_true_ids]).sum().item()
    # FN+FP
    fn_plus_fp = (outs != labels).sum().item()
    return tp / (tp + fn_plus_fp)


# %%
# モデルの学習
# =================================================
def train_model(net, epochs, dataloaders_dict, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    best_iou = 0.0
    loss_dict = {"train": [], "val": []}
    iou_dict = {"train": [], "val": []}
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1} / {epochs}")
        print("--------------------------")
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            pred_list = []
            true_list = []
            for images, labels in tqdm(dataloaders_dict[phase]):
                images = images.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * images.size(0)
                    preds = preds.to("cpu").numpy()
                    pred_list.extend(preds)
                    labels = labels.to("cpu").numpy()
                    true_list.extend(labels)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            tn, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
            epoch_iou = tp / (tp + fp + fn)
            loss_dict[phase].append(epoch_loss)
            iou_dict[phase].append(epoch_iou)
            print(f"{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}")
            if (phase == "val") and (epoch_iou > best_iou) and (epoch > 10):
                best_iou = epoch_iou
                param_name = f"./Epoch{epoch+1}_iou_{epoch_iou:.4f}.pth"
                torch.save(net.state_dict(), param_name)

    return loss_dict, iou_dict


# %%
net = preresnet.preresnet(depth=20)
net.conv1 = nn.Conv2d(
    7, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
)
net.fc = nn.Linear(in_features=64, out_features=2, bias=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

epochs = 150
loss_dict, iou_dict = train_model(
    net=net,
    epochs=epochs,
    dataloaders_dict=dataloaders_dict,
    loss_fn=loss_fn,
    optimizer=optimizer,
)




