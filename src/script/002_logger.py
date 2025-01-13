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
######################
# set dirs & filename
######################
serial_number = 10  # スプレッドシートAの番号

comp_name = "Satellite"

if "google.colab" in sys.modules:  # colab環境
    print("google.colab")
    INPUT_PATH = Path("/content")  # 読み込みファイル場所
    # name_notebook = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'] # ノートブック名を取得
    name_notebook = "exp008_size64_0109.ipynb"
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

train_dir = INPUT_PATH / "train/"  # 学習ファイルのフォルダ
pred_sub_dir = INPUT_PATH / "test/"  # testファイルのフォルダ

save_path = OUTPUT_EXP + f"/{name}_metrics.pkl"  # 保存と読み込みのパス

"""
os.pathとPathで連結の仕方が異なるので注意（そのうちもどちらかにする）
os.path 基本「+」でつなぐ
Path 「/」でつなげられる
"""

# %%

######################
# ハイパーパラメータの設定
# model_type:efficientnet_v2_s

num_workers = 2  # DataLoader CPU使用量

train_ratio = 0.9
test_ratio = 0.1

batch_size = 256
epochs = 30


# optimizer_type = "RAdam"
lr = 5e-4  # Adam  0.001　SGD 0.005
# weight_decay = 5e-4
# momentum = 0.9

# 最終学習率
# scheduler :CosineAnnealingLR
eta_min = 5e-6  # 1e-5

# 不均衡データのために重みをつける
# 損失関数に重みをつける
# weights =[total / len(pos), total / len(neg)]　# クラスごとの重みを各クラスのサンプル数に基づいて調整
weights = [23.8050, 1.0438]


# %%
# ログ保存　 stdout と stderr をリダイレクトするクラス
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


"""vscode"""
# utils
# # ログファイルの設定
logging.basicConfig(
    filename=f"{OUTPUT_EXP}/log_{name}.txt", level=logging.INFO, format="%(message)s"
)
# # ロガーの作成
logger = logging.getLogger()

#%%
# ログの保存
# stdout と stderr を一時的にリダイレクト
stdout_logger = logging.getLogger("STDOUT")
stderr_logger = logging.getLogger("STDERR")

sys_stdout_backup = sys.stdout
sys_stderr_backup = sys.stderr

sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

print("-" * 10, "result", "-" * 10)
# 評価値
df_metrics = pd.DataFrame(metrics, columns=["target", "nfold", "mae"])
print(f"MCMAE:{df_metrics['mae'].mean():.4f}")

# validの推論値
df_valid_pred_all = pd.pivot_table(
    df_valid_pred,
    index=[
        "engagementMetricsDate",
        "playerId",
        "date_playerId",
        "date",
        "yearmonth",
        "playerForTestSetAndFuturePreds",
    ],
    columns=["nfold"],
    values=list(df_valid_pred.columns[df_valid_pred.columns.str.contains("target")]),
    aggfunc=np.sum,
)
df_valid_pred_all.columns = [
    "{}_fold{}_{}".format(i.split("_")[0], j, i.split("_")[1])
    for i, j in df_valid_pred_all.columns
]
df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)

# リダイレクトを解除
sys.stdout = sys_stdout_backup
sys.stderr = sys_stderr_backup

#%%
"""Colab（出力遅い場合mount解除したら同期される）"""
#==========================================

OUTPUT_EXP = os.path.join(OUTPUT, name)  # logなど情報保存場所

log_file_path =  f"{OUTPUT_EXP}/log_{name}.txt"


# ロガー作成
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 既存のハンドラを削除
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# ファイルハンドラを追加  ログをファイルに保存するためのハンドラを設定
file_handler = logging.FileHandler(log_file_path, mode='a')#"a"追記モード
file_handler.setLevel(logging.INFO)

# コンソールハンドラの設定  コンソールにログを出力するためのハンドラを設定
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


# フォーマッターを設定 フォーマットを設定
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)


# ロガーの作成
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)



# ログを出力して確認
logger.info("This is a test log message.")


#%%
# 出力方法1
logger.info("-" * 10 +  "test1 " + "-" * 10)
logger.info("-" * 10 + " test2 " + "-" * 10)
logger.info(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))
#%%
#出力方法2

# stdout と stderr を一時的にリダイレクト
stdout_logger = logging.getLogger("STDOUT")
stderr_logger = logging.getLogger("STDERR")

sys_stdout_backup = sys.stdout
sys_stderr_backup = sys.stderr

sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)


"""
ここに以下のようにprintで出力
"""
# 終了時刻
print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))
print("-" * 10 +  "test7 " + "-" * 10)

# リダイレクトを解除
sys.stdout = sys_stdout_backup
sys.stderr = sys_stderr_backup



#%%
"""テンポラリ領域にメモを作成する方法"""
#=======================================================
# ログファイルの保存場所を変更
OUTPUT_EXP = "/content/logs"
os.makedirs(OUTPUT_EXP, exist_ok=True)

logging.basicConfig(
    filename=f"{OUTPUT_EXP}/log_{name}.txt", level=logging.INFO, format="%(message)s"
)

#%%
# OUTPUT_EXP = os.path.join(OUTPUT, name)  # logなど情報保存場所

OUTPUT_EXP = "/content/logs"
log_file_path = f"{OUTPUT_EXP}/log_{name}.txt"


# ロガー作成
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 既存のハンドラを削除
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# ファイルハンドラを追加  ログをファイルに保存するためのハンドラを設定
file_handler = logging.FileHandler(log_file_path, mode='a')#"a"追記モード
file_handler.setLevel(logging.INFO)

# コンソールハンドラの設定  コンソールにログを出力するためのハンドラを設定
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


# フォーマッターを設定 フォーマットを設定
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)


# ロガーの作成
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ログを出力して確認
logger.info("This is a test log message.")

#%%
# ログ作成
logger.info("-" * 10 +  "test1 " + "-" * 10)
logger.info("-" * 10 + " test2 " + "-" * 10)
logger.info(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

#%%
# ファイルをドライブに保存

# コピー元とコピー先のパスを指定
source_path = "/content/logs/log_{name}.txt""
destination_path = f"{OUTPUT_EXP}/log_{name}.txt"
OUTPUT_EXP = os.path.join(OUTPUT, name)  # logなど情報保存場所

# 保存先ディレクトリがない場合は作成
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# ファイルをコピー
shutil.copy(source_path, destination_path)


#%%
"""clabだと時間がかかる？"""
# # stdout と stderr を一時的にリダイレクト
# stdout_logger = logging.getLogger("STDOUT")
# stderr_logger = logging.getLogger("STDERR")

# sys_stdout_backup = sys.stdout
# sys_stderr_backup = sys.stderr

# sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
# sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)



# # 終了時刻
# print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))
# print("-" * 10 +  "test7 " + "-" * 10)

# # リダイレクトを解除
# sys.stdout = sys_stdout_backup
# sys.stderr = sys_stderr_backup

#%%
#その他
#ファイルが作れないときはこれで作成してみる
# 明示的にテキストファイルを作成してみる
log_file_path = f"{OUTPUT_EXP}/log_{name}.txt"

try:
    with open(log_file_path, "w") as test_file:
        test_file.write("This is a test log.")
    print("Log file created successfully:", log_file_path)
except Exception as e:
    print("Error creating log file:", e)