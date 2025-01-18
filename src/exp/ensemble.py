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


import numpy as np
import pandas as pd
from scipy import ndimage



#%%
######################
# set dirs & filename
######################
serial_number = 18  # スプレッドシートAの番号

comp_name = "Satellite"

if 'google.colab' in sys.modules:  # colab環境
    print("google.colab")
    INPUT_PATH = Path("/content")  # 読み込みファイル場所
    # name_notebook = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'] # ノートブック名を取得
    name_notebook = "exp101_Efv2M_0116.ipynb"
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



#%% 

# ファイルを読み込む
model1_predictions = pd.read_csv(f'{OUTPUT}/submission/exp012_size96_0113_sub28_05.tsv', sep='\t', header=None)
model2_predictions = pd.read_csv(f'{OUTPUT}/submission/exp101_Efv2M_0116_sub23_05.tsv', sep='\t', header=None)
# model3_predictions = pd.read_csv(f'{OUTPUT}/submission/model3_sub.tsv', sep='\t', header=None)

# 予測値部分だけ取得
pred1 = model1_predictions[1]
pred2 = model2_predictions[1]
# pred3 = model3_predictions[1]
#%%
print(pred1.head())
#%%

#平均0.5以上なら1
# ensemble_predictions = ((pred1 + pred2) / 2).astype(int)
# ensemble_predictions = (ensemble_predictions >= 0.5).astype(int)
#もしくは
ensemble_predictions = ((pred1+pred2)>=1).astype(int)


# 完全一致
# ensemble_predictions = ((pred1 == 1) & (pred2 == 1)).astype(int)


#%%
# アンサンブル結果を保存
submit_df = model1_predictions.copy()
submit_df[1] = ensemble_predictions
#%%
submit_df.to_csv(f'{OUTPUT}/submission/exp301_ensemble_sub.tsv', sep='\t', header=None, index=None)

#%%
#APIで提出
# ! signate submit --competition-id=263 "C:\Users\Takashi Hao\Desktop\Working\Torch_project\src\output\submission\exp301_ensemble_sub.tsv" --note "アンサンブルSとM or"
#%%
submit_df.head()

# %%
submit_df[1].mean()
# %%
len(submit_df[1])
# %%
