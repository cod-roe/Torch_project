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
# import json
import logging

# import re
import os
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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset


from sklearn.model_selection import StratifiedKFold, train_test_split  # , KFold
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
# name = 'run001'

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
warnings.simplefilter('ignore')

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
#%%
# マスターデータ読み込み
# =================================================
train_master = pd.read_csv(INPUT_PATH + "train_master.tsv", sep="\t")

sample_submit = pd.read_csv(INPUT_PATH +"sample_submit.tsv", sep="\t",header=None)

#%%
train_master.head()
# %%
pos = train_master.loc[train_master["flag"] == 1]
neg = train_master.loc[train_master["flag"] == 0]
print(f'学習データ件数: {len(train_master)}')
print(f'正例件数: {len(pos)}')
print(f'負例件数: {len(neg)}')
print(f'正例割合: {round(len(pos)/len(train_master), 3)}')
# %%
