# %% [markdown]
## EDA！
# =================================================
# データの確認
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
import pickle
from tqdm import tqdm_notebook as tqdm
import warnings
# import zipfile

import numpy as np
import pandas as pd


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


from sklearn.model_selection import train_test_split  # StratifiedKFold , KFold
# from sklearn.metrics import mean_squared_error,accuracy_score, roc_auc_score ,confusion_matrix

# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 0  # スプレッドシートAの番号

######################
# Data #
######################
comp_name = "Satellite"
# 評価：IOU 回帰 分類

skip_run = False  # 飛ばす->True，飛ばさない->False

######################
# filename
######################
# vscode用
abs_path = os.path.abspath(__file__)  # /tmp/work/src/exp/_.py'
name = os.path.splitext(os.path.basename(abs_path))[0]
# Google Colab等用（取得できないためファイル名を入力）
# name = 'baseline_1219'

######################
# set dirs #
######################
DRIVE = os.path.dirname(os.getcwd())  # このファイルの親(scr)
INPUT_PATH = f"../input/{comp_name}/"  # 読み込みファイル場所
OUTPUT = os.path.join(DRIVE, "output")
OUTPUT_EXP = os.path.join(OUTPUT, name)  # 情報保存場所
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")  # 学習済みモデル保存

######################
# Dataset #
######################

######################
# ハイパーパラメータの設定
######################


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
# マスターデータ読み込み
# =================================================
train_master = pd.read_csv(INPUT_PATH + "train_master.tsv", sep="\t")

sample_submit = pd.read_csv(INPUT_PATH + "sample_submit.tsv", sep="\t", header=None)

# %%
train_master.head()
# %%
pos = train_master.loc[train_master["flag"] == 1]
neg = train_master.loc[train_master["flag"] == 0]
print(f"学習データ件数: {len(train_master)}")
print(f"正例件数: {len(pos)}")
print(f"負例件数: {len(neg)}")
print(f"正例割合: {round(len(pos)/len(train_master), 3)*100}%")
# %%
sample_submit.head()
# %%
# 円グラフの表示
# =================================================

plt.figure(figsize=(7, 7))
plt.pie(x=[len(pos), len(neg)], labels=["Positive", "Negative"], colors=["red", "blue"])
plt.legend()
plt.title("正例と負例の割合")
plt.show()


# %%
def get_dims(file):
    "Returns dimenstions for an RBG image"
    im = Image.open(file)
    arr = np.array(im)
    h, w, d = arr.shape
    return h, w, d


# %%
# 学習データからランダムで1枚取り出して表示
# =================================================

train_size = len(train_master)  # 学習データのサイズ
idx = random.randint(0, train_size - 1)  # 0から学習データ数の範囲でランダムで整数を抽出
file = train_master["file_name"][idx]  # 画像ファイル名
label = train_master["flag"][idx]  # 画像ラベル
img_path = f"../input/Satellite/train/{file}"  # 画像が格納されているパス
img = io.imread(img_path)  # 画像を開く
print(f"画像形状：{img.shape}")

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
# 正例からランダムでサンプリングして画像を表示
# =================================================

idx = random.choice(pos.index.to_list())
file = train_master["file_name"][idx]
img_path = f"../input/Satellite/train/{file}"
img = io.imread(img_path)

# チャネルごとに表示
channel_label = ["1", "B", "G", "R", "5", "6", "7"]
print("ラベル: 正例")
fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(15, 5))
for i in range(7):
    ax[i].imshow(img[:, :, i])
    ax[i].set_title(channel_label[i])
    ax[i].set_axis_off()
plt.show()
# %%
# 負例からランダムでサンプリングして画像を表示
# =================================================

idx = random.choice(neg.index.to_list())
file = train_master["file_name"][idx]
img_path = f"../input/Satellite/train/{file}"
img = io.imread(img_path)

# チャネルごとに表示
channel_label = ["1", "B", "G", "R", "5", "6", "7"]
print("ラベル: 負例")
fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(15, 5))
for i in range(7):
    ax[i].imshow(img[:, :, i])
    ax[i].set_title(channel_label[i])
    ax[i].set_axis_off()
plt.show()
# %%
# 正例を複数表示
# =================================================

sample_ids = random.sample(pos.index.to_list(), 3)

# 表示
print("正例サンプル")
fig, ax = plt.subplots(nrows=3, ncols=7, figsize=(15, 5))
for i, idx in enumerate(sample_ids):
    file = train_master["file_name"][idx]
    img_path = f"../input/Satellite/train/{file}"
    img = io.imread(img_path)
    for j in range(7):
        ax[i, j].imshow(img[:, :, j])
        ax[i, j].set_axis_off()
        if i == 0:
            ax[i, j].set_title(channel_label[j])
        if j == 0:
            ax[i, j].set_ylabel(f"idx: {idx}")
plt.show()
# %%
# 負例を複数表示
# =================================================

sample_ids = random.sample(neg.index.to_list(), 3)

# 表示
print("負例サンプル")
fig, ax = plt.subplots(nrows=3, ncols=7, figsize=(15, 5))
for i, idx in enumerate(sample_ids):
    file = train_master["file_name"][idx]
    img_path = f"../input/Satellite/train/{file}"
    img = io.imread(img_path)
    for j in range(7):
        ax[i, j].imshow(img[:, :, j])
        ax[i, j].set_axis_off()
        if i == 0:
            ax[i, j].set_title(channel_label[j])
plt.show()
# %%
# 正例チャネル別ヒストグラム(単一データ)
# =================================================

idx = random.choice(pos.index.to_list())
file = train_master["file_name"][idx]
img_path = f"../input/Satellite/train/{file}"
pos_img = io.imread(img_path)

print("正例サンプル")
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 15))
for i in range(7):
    if i % 2 == 0:
        ax[i // 2, 0].hist(pos_img[:, :, 0].ravel(), bins=100)
        ax[i // 2, 0].set_title(channel_label[i])
        ax[i // 2, 0].set_axis_off()
    else:
        ax[i // 2, 1].hist(pos_img[:, :, 0].ravel(), bins=100)
        ax[i // 2, 1].set_title(channel_label[i])
        ax[i // 2, 1].set_axis_off()
plt.show()
# %%
# 負例チャネル別ヒストグラム(単一データ)
idx = random.choice(neg.index.to_list())
file = train_master["file_name"][idx]
img_path = f"../input/Satellite/train/{file}"
neg_img = io.imread(img_path)

print("負例サンプル")
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 15))
for i in range(7):
    if i % 2 == 0:
        ax[i // 2, 0].hist(neg_img[:, :, 0].ravel(), bins=100)
        ax[i // 2, 0].set_title(channel_label[i])
        ax[i // 2, 0].set_axis_off()
    else:
        ax[i // 2, 1].hist(neg_img[:, :, 0].ravel(), bins=100)
        ax[i // 2, 1].set_title(channel_label[i])
        ax[i // 2, 1].set_axis_off()
plt.show()
# %%
# 正例チャネル別統計量(単一データ)
for i in range(7):
    tmp = pos_img[:, :, i].ravel()
    print(f"{channel_label[i]}")
    print(f"max: {tmp.max()}")
    print(f"min: {tmp.min()}")
    print(f"std: {round(tmp.std(),2)}")
    print(f"mean: {round(tmp.mean(),2)}")
    print("\n---------------------------")
# %%
# 負例チャネル別統計量(単一データ)
for i in range(7):
    tmp = neg_img[:, :, i].ravel()
    print(f"{channel_label[i]}")
    print(f"max: {tmp.max()}")
    print(f"min: {tmp.min()}")
    print(f"std: {round(tmp.std(),2)}")
    print(f"mean: {round(tmp.mean(),2)}")
    print("\n---------------------------")


# %%
# 関数定義
def get_stats(x_array, round_num=2):
    """
    ndarrayから得られる統計量をPythonのlist型で返す
    """
    # 次元によらない統計量を出す
    if x_array.ndim != 1:
        x_array = x_array.ravel()

    max = x_array.max()
    min = x_array.min()
    std = round(x_array.std(), round_num)
    mean = round(x_array.mean(), round_num)
    percentile_1 = np.percentile(x_array, 1)
    percentile_5 = np.percentile(x_array, 5)
    percentile_25 = np.percentile(x_array, 25)
    percentile_75 = np.percentile(x_array, 75)
    percentile_95 = np.percentile(x_array, 95)
    percentile_99 = np.percentile(x_array, 99)

    return [
        max,
        min,
        std,
        mean,
        percentile_1,
        percentile_5,
        percentile_25,
        percentile_75,
        percentile_95,
        percentile_99,
    ]


# %%
# DataFrameで可視化
pos_channel_data = [get_stats(pos_img[:, :, i]) for i in range(7)]
pos_stats = pd.DataFrame(
    data=pos_channel_data,
    columns=["max", "min", "std", "mean", "1%", "5%", "25%", "75%", "95%", "99%"],
    index=channel_label,
)
neg_channel_data = [get_stats(neg_img[:, :, i]) for i in range(7)]
neg_stats = pd.DataFrame(
    data=neg_channel_data,
    columns=["max", "min", "std", "mean", "1%", "5%", "25%", "75%", "95%", "99%"],
    index=channel_label,
)
print("正例データ統計量(サンプル数1)")
display(pos_stats)
print("\n負例データ統計量(サンプル数1)")
display(neg_stats)
# %%
# ランダムでサンプリング
idx = random.choice(pos.index.to_list())
# idx = random.choice(neg.index.to_list())
file = train_master["file_name"][idx]
img_path = f"../input/Satellite/train/{file}"
img = io.imread(img_path)

# RGBチャネルを抜き出して(両端1%クリッピング&)正規化したうえで結合
r_channel = img[:, :, 3]
r_max, r_min = r_channel.ravel().max(), r_channel.ravel().min()
r_mean, r_std = r_channel.ravel().mean(), r_channel.ravel().std()
# r_max, r_min = np.percentile(r_channel.ravel(), 99), np.percentile(r_channel.ravel(), 1)
# r_channel = r_channel.clip(min=r_min, max=r_max)
r_channel_scaled = (r_channel - r_min) / (r_max - r_min)
r_channel_normalized = (r_channel - r_mean) / r_std
r_channel = r_channel[:, :, np.newaxis]
r_channel_scaled = r_channel_scaled[:, :, np.newaxis]
r_channel_normalized = r_channel_normalized[:, :, np.newaxis]

g_channel = img[:, :, 2]
g_max, g_min = g_channel.ravel().max(), g_channel.ravel().min()
g_mean, g_std = g_channel.ravel().mean(), g_channel.ravel().std()
# g_max, g_min = np.percentile(g_channel.ravel(), 99), np.percentile(g_channel.ravel(), 1)
# g_channel = g_channel.clip(min=g_min, max=g_max)
g_channel_scaled = (g_channel - g_min) / (g_max - g_min)
g_channel_normalized = (g_channel - g_mean) / g_std
g_channel = g_channel[:, :, np.newaxis]
g_channel_scaled = g_channel_scaled[:, :, np.newaxis]
g_channel_normalized = g_channel_normalized[:, :, np.newaxis]

b_channel = img[:, :, 1]
b_max, b_min = b_channel.ravel().max(), b_channel.ravel().min()
b_mean, b_std = b_channel.ravel().mean(), b_channel.ravel().std()
# b_max, b_min = np.percentile(b_channel.ravel(), 99), np.percentile(b_channel.ravel(), 1)
# b_channel = b_channel.clip(min=b_min, max=b_max)
b_channel_scaled = (b_channel - b_min) / (b_max - b_min)
b_channel_normalized = (b_channel - b_mean) / b_std
b_channel = b_channel[:, :, np.newaxis]
b_channel_scaled = b_channel_scaled[:, :, np.newaxis]
b_channel_normalized = b_channel_normalized[:, :, np.newaxis]

rgb_img = np.concatenate([r_channel, g_channel], axis=2)
rgb_img = np.concatenate([rgb_img, b_channel], axis=2)
rgb_img_scaled = np.concatenate([r_channel_scaled, g_channel_scaled], axis=2)
rgb_img_scaled = np.concatenate([rgb_img_scaled, b_channel_scaled], axis=2)
rgb_img_normalized = np.concatenate(
    [r_channel_normalized, g_channel_normalized], axis=2
)
rgb_img_normalized = np.concatenate([rgb_img_normalized, b_channel_normalized], axis=2)
rgb_img_normalized = np.clip(rgb_img_normalized, 0, 1)

# 画像表示
# print('ラベル: 正例')
plt.figure(figsize=(2, 2))
plt.imshow(rgb_img_scaled)
plt.title("正例サンプル(正規化)")
plt.axis("off")
plt.show()
plt.figure(figsize=(2, 2))
plt.imshow(rgb_img_normalized)
plt.title("正例サンプル(標準化)")
plt.axis("off")
plt.show()
# %%
# 前処理
# =================================================
# (H,W,C)から(C,H,W)へ変換
pos_img = pos_img.transpose(2, 0, 1)
pos_img.shape
# %%
# テンソル変換
pos_img_torch = torch.from_numpy(pos_img.astype(np.float32)).clone()
pos_img_torch.shape
# %%
# 正規化
norm = transforms.Normalize(
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
)
norm(pos_img_torch)
# %%
# 中央くり抜き
centercrop = transforms.CenterCrop(16)
centercrop(pos_img_torch)


# %%
# パイプライン作成
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
# パイプライン使用（今回はなし）
"""plt.imshow(rgb_img_scaled)
plt.title('変換前(正規化して表示)')
plt.show()

rgb_img_trans = rgb_img
#rgb_img_trans = rgb_img.transpose(2, 0, 1)
mean = (r_mean, g_mean, b_mean)
std = (r_std, g_std, b_std)
transform = ImageTransform(mean, std)
rgb_img_trans = transform(rgb_img_trans, phase='train')
rgb_img_trans = rgb_img_trans.numpy().transpose((1, 2, 0))
rgb_img_trans = np.clip(rgb_img_trans, 0, 1)
plt.imshow(rgb_img_trans)
plt.title('変換後')
plt.show()"""
# %%
# データセット作成
# =================================================
train_dir = "../input/Satellite/train/"


class AistDataset(Dataset):
    def __init__(
        self, dir, file_list, min, max, transform=None, phase="train", channel="RGB"
    ):
        """
        衛星画像の学習用データセット

        Attributes
        -------------------
        dir : str
            画像が保管されているパス
        file_list : dataframe
            画像のファイル名とフラグが格納されているデータフレーム
        min : int
            画素値の最小値(クリッピング用)
        max : int
            画素値の最大値(クリッピング用)
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
        self.min = min
        self.max = max

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # 画像をロード
        img_path = self.image_path[idx]
        img = io.imread(self.dir + img_path)
        img = np.clip(img, self.min, self.max)
        img = (img - self.min) / (self.max - self.min)
        # RGB指定があれば次元を限定する
        if self.channel == "RGB":
            img = img[:, :, 1:4]  # BGR
        # 前処理の実装
        if self.transform:
            img = self.transform(img, self.phase)
        label = self.image_label[idx]

        return img, label


# %%
"""# 標準化のための平均値と標準偏差を計算
# メモリの都合上10万枚ごとにまとめる

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tmp_set = 1
cnt = 1
# 1.train_masterからファイルパスを抽出
for i, file in enumerate(train_master['file_name'].to_list()):
    # 2.ndarrayで読み込み
    tmp = io.imread(f'./train/{file}')
    tmp = tmp[np.newaxis, :, :, :] #(H,W,C)で読み込まれるため(B,H,W,C)に変換
    # 3.tensorに変換
    tmp = torch.from_numpy(tmp.astype(np.float32)).clone()
    # 4.cudaに移す
    tmp = tmp.to(device)
    # 5.次元拡張
    #tmp = torch.unsqueeze(tmp, dim=0)
    # 6.データ結合
    if i == 0:
        train_data = tmp
        train_data = train_data.to(device)
    elif tmp_set != cnt:
        train_data = tmp
        train_data = train_data.to(device)
        tmp_set = cnt
    else:
        train_data = torch.cat([train_data, tmp], dim=0)
    if (i+1) % 10000 == 0:
        print(f'{i+1}枚目')
    if (i+1) % 100000 == 0 or i == len(train_master)-1:
        # 7.ndarrayに変換
        train_data = train_data.to('cpu').detach().numpy().copy()
        # 8.保存
        path = f'/content/drive/MyDrive/Signate/AIST/ndarray/train_{cnt}'
        np.save(path, train_data)
        cnt += 1

# 9.読み込みと結合
train_data_1 = np.load('/content/drive/MyDrive/Signate/AIST/ndarray/train_1.npy')
train_data_2 = np.load('/content/drive/MyDrive/Signate/AIST/ndarray/train_2.npy')
train_data_3 = np.load('/content/drive/MyDrive/Signate/AIST/ndarray/train_4.npy')
train_data = np.concatenate([train_data_1, train_data_2], axis=0)

# 10.形状は[B,H,W,C]となっているためチャネルごと(4次元)の平均と標準偏差を算出する
band_1_mean = np.mean(train_data[:, :, :, 0].flatten())
band_b_mean = np.mean(train_data[:, :, :, 1].flatten())
band_g_mean = np.mean(train_data[:, :, :, 2].flatten())
band_r_mean = np.mean(train_data[:, :, :, 3].flatten())
band_5_mean = np.mean(train_data[:, :, :, 4].flatten())
band_6_mean = np.mean(train_data[:, :, :, 5].flatten())
band_7_mean = np.mean(train_data[:, :, :, 6].flatten())
band_1_std = np.std(train_data[:, :, :, 0].flatten())
band_b_std = np.std(train_data[:, :, :, 1].flatten())
band_g_std = np.std(train_data[:, :, :, 2].flatten())
band_r_std = np.std(train_data[:, :, :, 3].flatten())
band_5_std = np.std(train_data[:, :, :, 4].flatten())
band_6_std = np.std(train_data[:, :, :, 5].flatten())
band_7_std = np.std(train_data[:, :, :, 6].flatten())
bgr_mean = (band_b_mean, band_g_mean, band_r_mean)
bgr_std = (band_b_std, band_g_std, band_r_std)"""
# %%
# 統計値の可視化: 1%と99%付近の値(min3600,max23500くらいでクリッピング)
"""train_channel_data = [get_stats(train_data[:,:,i]) for i in range(7)]
train_stats = pd.DataFrame(data=train_channel_data,
                         columns=['max', 'min', 'std', 'mean', '1%', '5%', '25%', '75%', '95%', '99%'],
                         index=channel_label)
display(train_stats)"""
# %%
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

train_dataset = AistDataset(
    dir=train_dir,
    file_list=train_data,
    min=3600,
    max=23500,
    transform=ImageTransform(),
    phase="train",
    channel="RGB",
)
valid_dataset = AistDataset(
    dir=train_dir,
    file_list=valid_data,
    min=3600,
    max=23500,
    transform=ImageTransform(),
    phase="val",
    channel="RGB",
)
eval_dataset = AistDataset(
    dir=train_dir,
    file_list=eval_data,
    min=3600,
    max=23500,
    transform=ImageTransform(),
    phase="val",
    channel="RGB",
)

# %%
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
# ResNet読み込みとハイパーパラメータの設定
# =================================================

# %%
# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# %%
# ResNet50の読み込み
model = models.resnet50(pretrained=True)
# model = models.googlenet(pretrained=True)
model = model.to(device)
# %%
# ハイパーパラメータの設定
num_epoch = 25
lr = 0.005
batch_size = 1024
train_ratio = 0.75
weight_decay = 5e-4
momentum = 0.9

# 最適化アルゴリズムと損失関数の設定
optimizer = optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()


# 評価指標IoUの設計(pytorchのtensorを想定)
def IoU(predicts, labels):
    outs = predicts.max(1)[1]
    # TP
    p_true_ids = torch.nonzero(outs)  # 正例と予測したインスタンスのインデックス番号
    tp = (outs[p_true_ids] == labels[p_true_ids]).sum().item()
    # FN+FP
    fn_plus_fp = (outs != labels).sum().item()
    return tp / (tp + fn_plus_fp)


# DataLoader
trainloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
validloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
# %%
model
# %%
# タスクに合わせてレイヤを調整
model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
model = model.to(device)
# %%
# 学習
# =================================================
train_loss_list = []
train_iou_list = []
valid_loss_list = []
valid_iou_list = []

for epoch in range(num_epoch):
    train_loss = 0
    train_iou = 0
    valid_loss = 0
    valid_iou = 0

    # 訓練モード
    model.train()
    for i, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.float()
        imgs, labels = imgs.to(device), labels.to(device)
        # 勾配のリセット
        optimizer.zero_grad()
        # 出力
        outputs = model(imgs)
        # 損失計算
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        # ioc
        train_iou += IoU(outputs, labels)
        # 逆伝播
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / len(trainloader.dataset)
    avg_train_iou = train_iou / (i + 1)

    # 検証モード
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(validloader):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.float()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            valid_iou += IoU(outputs, labels)
    avg_val_loss = valid_loss / len(validloader.dataset)
    avg_val_iou = valid_iou / (i + 1)
    print(
        "Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, train_iou: {train_iou:.4f}, val_iou: {val_iou:.4f}".format(
            epoch + 1,
            num_epoch,
            i + 1,
            loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_iou=avg_train_iou,
            val_iou=avg_val_iou,
        )
    )
    tmp_loss_list = valid_loss_list
    train_loss_list.append(avg_train_loss)
    train_iou_list.append(avg_train_iou)
    valid_loss_list.append(avg_val_loss)
    valid_iou_list.append(avg_val_iou)
# %%
# モデルの保存と読み込み
model_path = EXP_MODEL + f"/{name}_resnet50.pth"
torch.save(model.state_dict(), model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50()
model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
# %%
# 評価用データで精度確認
# DataLoaderを使う
evalloader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
eval_loss = 0
eval_iou = 0
eval_iou_list = []
model.eval()
with torch.no_grad():
    for i, (imgs, labels) in enumerate(evalloader):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = imgs.float()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        eval_loss += loss.item()
        eval_iou += IoU(outputs, labels)
        eval_iou_list.append(IoU(outputs, labels))
avg_eval_loss = eval_loss / len(evalloader.dataset)
avg_eval_iou = eval_iou / (i + 1)
print(avg_eval_iou)
# %%
# 1枚ずつ推論
min, max = 3600, 23500
model.eval()
for i, file in enumerate(eval_files.values):
    # 1.ndarrayで読み込み
    img = io.imread(f"../input/Satellite/train/{file}")
    # 2. 正規化
    img = np.clip(img, min, max)
    img = (img - min) / (max - min)
    # 3. チャネル絞り込み
    img = img[:, :, 1:4]
    # 4. 前処理
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :, :, :]
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.to(device)
    # 5. 推論
    output = model(img)
    # 6. ラベルにしてndarrayに変換
    output = output.max(1)[1].to("cpu").detach().numpy().copy()
    # 7. 結合
    if i == 0:
        prediction = output
    else:
        prediction = np.concatenate([prediction, output], axis=0)

# 8. IoU算出
eval_df = pd.DataFrame(
    data=[
        [f, pred, label]
        for f, pred, label in zip(eval_files.values, prediction, eval_labels.values)
    ],
    columns=["file_name", "prediction", "label"],
)

tp = len(eval_df.loc[(eval_df["prediction"] == 1) & (eval_df["label"] == 1)])
fp = len(eval_df.loc[(eval_df["prediction"] == 1) & (eval_df["label"] == 0)])
fn = len(eval_df.loc[(eval_df["prediction"] == 0) & (eval_df["label"] == 1)])
iou = tp / (tp + fp + fn)
print(iou)
#%%
# 提出形式を確認
sample_submit.columns = ['file', 'prediction']
sample_submit
#%%
# 1枚ずつ推論

min, max = 3600, 23500
model.eval()
for i, file in enumerate(sample_submit['file'].to_list()):
    # 1.ndarrayで読み込み
    img = io.imread(f'../input/Satellite/test/{file}')
    # 2. 正規化
    img = np.clip(img, min, max)
    img = (img - min) / (max - min)
    # 3. チャネル絞り込み
    #img = img[:, 1:4, :, :]
    img = img[:, :, 1:4]
    # 4. 前処理
    img = np.transpose(img, (2,0,1))
    img = img[np.newaxis, :, :, :]
    img = torch.from_numpy(img.astype(np.float32)).clone()
    img = img.to(device)
    # 5. 推論
    output = model(img)
    # 6. ラベルにしてndarrayに変換
    output = output.max(1)[1].to('cpu').detach().numpy().copy()
    # 7. 結合
    if i == 0:
        prediction = output
    else:
        prediction = np.concatenate([prediction, output], axis=0)
    if (i+1) % 50000 == 0:
        print(f'{i+1}枚目まで終了')

#%%
# 7-35:提出ファイルの作成
file_name = sample_submit['file'].values
submit_df = pd.DataFrame(data=[[f, pred] for f, pred in zip(file_name, prediction)])
submit_df.to_csv(os.path.join(OUTPUT_EXP,  f"{file_name}_resnet50.tsv"), sep='\t', header=None, index=None)
