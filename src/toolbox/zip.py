# %%
import zipfile
import os
import shutil  # colab


def extract_all(zip_name, extract_to="."):
    with zipfile.ZipFile(zip_name, "r") as zipf:
        zipf.extractall(extract_to)


# %%
# test_1解凍
extract_all("../input/Satellite/test_1.zip", "test")
# %%
# test_2解凍
extract_all("../input/Satellite/test_2.zip", "test")
# %%
# test_3解凍
extract_all("../input/Satellite/test_3.zip", "test")
# %%

# test_4解凍
extract_all("../input/Satellite/test_4.zip", "test")


# %%
# signateAPI用トークンをコピー
# ============================
# コピー元とコピー先のパスを指定
source_path = "/content/drive/MyDrive/Python/SIGNATE/signate.json"
destination_path = "/root/.signate/signate.json"

# 保存先ディレクトリがない場合は作成
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# ファイルをコピー
shutil.copy(source_path, destination_path)

print("signate.json copied successfully.")
# %%
# ファイル取得
#! signate download --competition-id=263 --file-id=224
#! signate download --competition-id=263 --file-id=225
#! signate download --competition-id=263 --file-id=226
#! signate download --competition-id=263 --file-id=231
#! signate download --competition-id=263 --file-id=232

# 解凍
#!unzip train_1.zip -d /content/
#!unzip train_2.zip -d /content/
#!unzip train_3.zip -d /content/

# 容量に空きを作るためフォルダ削除
shutil.rmtree("/content/train")

# ドライブからコピー
# !cp "/content/drive/MyDrive/Python/SIGNATE/Satellite/data/test_1.zip" .
# !cp "/content/drive/MyDrive/Python/SIGNATE/Satellite/data/test_2.zip" .
# !cp "/content/drive/MyDrive/Python/SIGNATE/Satellite/data/test_3.zip" .
# !cp "/content/drive/MyDrive/Python/SIGNATE/Satellite/data/test_4.zip" .


# signateAPIを使ってテストファイルダウンロード
#! signate download --competition-id=263 --file-id=227
#! signate download --competition-id=263 --file-id=228
# ! signate download --competition-id=263 --file-id=229
# ! signate download --competition-id=263 --file-id=230

# APIで提出
# ! signate submit --competition-id=263 /content/drive/MyDrive/Python/SIGNATE/Satellite/output/exp007_BrCont_0109/exp007_BrCont_0109_sub30.tsv --note 初期学習と追加学習

# %%
# その他
# ====================================
# %%
# ローカルからテンポラリ領域にアップロード
from google.colab import files

uploaded = files.upload()  # ファイルを選択してアップロード
# %%
# 解凍せずに読み込む
import zipfile
import cv2
import numpy as np

with zipfile.ZipFile("data.zip", "r") as zip_ref:
    with zip_ref.open("image1.jpg") as file:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
