# %% [markdown]
## モデルの変更！
# =================================================
"""
EfficientNetV2-S
7チャンネル使用
パイプライン自作
"""

# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

# import gc
# import glob
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

# img_path = f"../input/Satellite/train/{file}"  # 画像が格納されているパス
img_path = f"./train/{file}"  # 画像が格納されているパス

# image = io.imread(img_path)
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
# パイプライン

# %%
# 前処理
"""
1.クリッピング 
2.正規化  外れ値があるためクリッピング→正規化
3.Data Augmentation
    水平/垂直フリップ、回転など
4.テンソル変換とリサイズ:画像の形状を変更し、テンソル化・リサイズを行います。
"""


class ClipAndNormalize:
    # クリッピングと正規化
    def __init__(self, min_value=5000, max_value=30000):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image):
        image = np.clip(image, self.min_value, self.max_value)
        image = (image - self.min_value) / (self.max_value - self.min_value)
        return image


class Augmentation:
    def __call__(self, image):
        # print(f"入力画像形状最初: {image.shape}")  # デバッグ
        # 水平フリップ
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()  # axis=2は幅方向
            # print(f"入力画像形状水平フリップ後: {image.shape}")  # デバッグ

        # 垂直フリップ
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()  # axis=1は高さ方向
            # print(f"入力画像形状垂直フリップ後: {image.shape}")  # デバッグ

        #  ランダム回転
        p = random.random()  # 0.0 ~ 1.0 の乱数
        if p < 0.25:
            # 回転なし
            return image
        elif p < 0.5:
            # 90度回転
            image = ndimage.rotate(image, 90, axes=(1, 2), reshape=False).copy()
        elif p < 0.75:
            # 180度回転
            image = ndimage.rotate(image, 180, axes=(1, 2), reshape=False).copy()
        else:
            # 270度回転
            image = ndimage.rotate(image, 270, axes=(1, 2), reshape=False).copy()

        # print(f"入力画像形状回転後: {image.shape}")  # デバッグ
        return image


class ToTensorAndResize:
    # テンソル化、形状変換、リサイズ（EfficientNetV2-S仕様）
    def __init__(self, resize_size=(224, 224)):
        """
        Args:
            resize_size (tuple): リサイズ先のサイズ (height, width)。
        """
        self.resize_size = resize_size

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): 入力画像 (C, H, W) または (H, W, C)。

        Returns:
            torch.Tensor: 処理後のテンソル (C, resize_height, resize_width)。
        """
        # NumPy → テンソル変換
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
            if (
                image.ndim == 3 and image.shape[-1] != image.shape[0]
            ):  # (H, W, C) ->  (C, H, W)
                image = image.permute(2, 0, 1)

        # リサイズ処理
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.resize_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return image


# %%
class ImageTransform:
    def __init__(self):
        """
        transforms.Composeは不使用
        学習時と検証時で挙動を変える
        >Data Augmentationの処理は学習時にのみに適用
        """

        self.data_transform = {
            "train": [
                ClipAndNormalize(),
                Augmentation(),
                ToTensorAndResize(),
            ],
            "val": [
                ClipAndNormalize(),
                ToTensorAndResize(),
            ],
        }

    def __call__(self, image, phase="train"):
        for transform in self.data_transform[phase]:
            image = transform(image)
        return image


# %%
"""
# 入力画像（ダミー）
sample_image = np.random.randint(0, 65535, size=(32, 32, 7)).astype(np.float32)

# 変換器の準備
transformer = ImageTransform()

# 学習用前処理
train_image = transformer(sample_image, phase="train")

# バリデーション用前処理
val_image = transformer(sample_image, phase="val")

print(f"学習用画像形状: {train_image.shape}")  # torch.Tensor: (7, 224, 224)
print(f"バリデーション用画像形状: {val_image.shape}")  # torch.Tensor: (7, 224, 224)
"""


# %%
train_dir = INPUT_PATH / "train/"


class SatelliteDataset(Dataset):
    """
    1. __init__:初期化を行う。
    2. __len__:1エポックあたりに使用するデータ数を返す。
    3. __getitem__:データの読み込み、前処理を行った上で、入力画像と正解ラベルのセットを返す。

    """

    def __init__(self, dir, file_list, transform=None, phase="train"):
        """
        衛星画像の学習用データセット
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
        """
        self.dir = dir
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.image_path = file_list["file_name"].to_list()
        self.image_label = file_list["flag"].to_list()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # 画像をロード
        img_path = self.image_path[idx]
        image = io.imread(self.dir / img_path)
        # 前処理の実装
        if self.transform:
            image = self.transform(img, self.phase)
        label = self.image_label[idx]
        return image, label


# %%
# Datasetの作成
# 学習用、検証用、評価用に分割(デフォルトは6:2:2とする)


train_size = 0.6
test_size = 0.5

train_files, eval_files, train_labels, eval_labels = train_test_split(
    train_master["file_name"],
    train_master["flag"],
    test_size=(1 - train_size) * test_size,
    stratify=train_master["flag"],
    random_state=0,
)

# valid_files, eval_files, valid_labels, eval_labels = train_test_split(valid_files, valid_labels, test_size=test_size, stratify=valid_labels, random_state=0)
# ここで正例と負例の割合を1:1にするアンダーサンプリング

train_files, valid_files, train_labels, valid_labels = train_test_split(
    train_files,
    train_labels,
    test_size=test_size * train_size,
    stratify=train_labels,
    random_state=0,
)


print(f"学習用データサイズ: {len(train_files)}")
print(f"検証用データサイズ: {len(valid_files)}")
print(f"評価用データサイズ: {len(eval_files)}")
print(f"学習用データ正例比率: {train_labels.mean()}")
print(f"検証用データ正例比率: {valid_labels.mean()}")
print(f"評価用データ正例比率: {eval_labels.mean()}")

train_data = train_master.loc[train_master.index.isin(train_files.index)].reset_index(
    drop=True
)
valid_data = train_master.loc[train_master.index.isin(valid_files.index)].reset_index(
    drop=True
)
eval_data = train_master.loc[train_master.index.isin(eval_files.index)].reset_index(
    drop=True
)

train_dataset = SatelliteDataset(
    dir=train_dir, file_list=train_data, transform=ImageTransform(), phase="train"
)

valid_dataset = SatelliteDataset(
    dir=train_dir, file_list=valid_data, transform=ImageTransform(), phase="val"
)

eval_dataset = SatelliteDataset(
    dir=train_dir, file_list=eval_data, transform=ImageTransform(), phase="val"
)

# %%
# 確認
# それぞれのデータセットからランダムサンプリングを行い挙動を確認
tr_idx = random.randint(0, len(train_dataset))
val_idx = random.randint(0, len(valid_dataset))
eval_idx = random.randint(0, len(eval_dataset))

tr_sample = train_dataset[tr_idx]
val_sample = valid_dataset[val_idx]
eval_sample = eval_dataset[eval_idx]

print(f"学習用データサンプル形状: {tr_sample[0].shape}")
print(f"学習用データサンプルラベル: {tr_sample[1]}")
print(f"検証用データサンプル形状: {val_sample[0].shape}")
print(f"検証用データサンプルラベル: {val_sample[1]}")
print(f"評価用データサンプル形状: {eval_sample[0].shape}")
print(f"評価用データサンプルラベル: {eval_sample[1]}")

# %%
# DataLoaderの作成
# =================================================
# DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

dataloaders_dict = {"train": train_dataloader, "val": valid_dataloader}
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

# モデルの出力層の再定義
model.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# デバイスに移動
model = model.to(device)

# %%
# 最適化アルゴリズムと損失関数の設定
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


# %%
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
# モデルの確認
model

# %%
# モデルの学習の関数
# =================================================


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
            tn, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
            epoch_iou = tp / (tp + fp + fn)
            loss_dict[phase].append(epoch_loss)
            iou_dict[phase].append(epoch_iou)
            print(f"{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}")
            if (phase == "val") and (epoch_iou > best_iou) or ((epoch + 1) == epochs):
                best_iou = epoch_iou
                checkpoint_path = (
                    EXP_MODEL / f"{name}_epoch{epoch+1}_iou_{epoch_iou:.4f}.pth"
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"Model checkpoint saved at {checkpoint_path}")

    return loss_dict, iou_dict


# %%



# %%
# モデルの学習
start_epoch = 0
stop_epoch = 5

loss_dict, iou_dict = train_model(
    model=model,
    start_epoch=start_epoch,
    stop_epoch=stop_epoch,
    epochs=epochs,
    dataloaders_dict=dataloaders_dict,
    criterion=criterion,
    optimizer=optimizer,
)
# %%

# DataLoaderを使う

evalloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
)


eval_loss = 0.0
pred_list = []
true_list = []
model.eval()
with torch.no_grad():
    for images, labels in tqdm(evalloader):
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

epoch_loss = eval_loss / len(evalloader.dataset)
tn, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
eval_iou = tp / (tp + fp + fn)
print(eval_iou)

