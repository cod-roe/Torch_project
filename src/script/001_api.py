#%%
#!pip install signate
#%%
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
from requests import get #colab
import shutil #colab
from tqdm import tqdm_notebook as tqdm
import warnings
# import zipfile

#%%
#signateAPI用トークンをコピー

# コピー元とコピー先のパスを指定
source_path = '/content/drive/MyDrive/Python/SIGNATE/signate.json'
destination_path = '/root/.signate/signate.json'

# 保存先ディレクトリがない場合は作成
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# ファイルをコピー
shutil.copy(source_path, destination_path)

print("signate.json copied successfully.")
#%%
# ! signate list

#ファイル表示
#! signate files --competition-id=263

#%%
#ファイル取得
# ! signate download --competition-id=263 --file-id=224
# ! signate download --competition-id=263 --file-id=225
# ! signate download --competition-id=263 --file-id=226
# ! signate download --competition-id=263 --file-id=231
# ! signate download --competition-id=263 --file-id=232

#%%
#解凍
# !unzip train_1.zip -d /content/
# !unzip train_2.zip -d /content/
# !unzip train_3.zip -d /content/

#%%
#APIで提出
# ! signate submit --competition-id=263 /content/drive/MyDrive/Python/SIGNATE/Satellite/output/exp008_size64_0109/exp008_size64_0109_sub30.tsv --note 初期学習と追加学習