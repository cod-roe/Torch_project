#%%gptk.ver
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

from PIL import Image



#%%
# 1. 前処理設計（Transforms）

# 前処理を設計
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # サイズ変更
    transforms.ToTensor(),          # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])
#もしくは
class ImagePreprocessor:
    def __init__(self, resize=(224, 224)):
        self.resize = resize

    def __call__(self, image):
        """
        画像をリサイズする処理
        """
        if not isinstance(image, Image.Image):  # PIL Imageであることを確認
            raise ValueError("Input must be a PIL.Image.Image object")
        return image.resize(self.resize)
#%%
# 2. Datasetの作成
"""
1. __init__:初期化を行う。
2. __len__:1エポックあたりに使用するデータ数を返す。
3. __getitem__:データの読み込み、前処理を行った上で、入力画像と正解ラベルのセットを返す。
"""
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])  # 画像を読み込む
        label = self.labels[idx]  # ラベルを取得

        if self.transform:
            image = self.transform(image)  # 前処理を適用

        return image, label

#%%
# 3. DataLoaderの作成

# CustomDatasetをインスタンス化
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # 画像のパス
labels = [0, 1, 0]  # ラベル
dataset = CustomDataset(image_paths, labels, transform=transform)

# DataLoaderを作成
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


#4. モデルの定義


# EfficientNetV2-Sのモデルを定義
model = models.efficientnet_v2_s(weights=None)

# 最初の層を変更（7チャネル対応）
original_conv = model.features[0][0]
model.features[0][0] = nn.Conv2d(
    in_channels=7,  # 7チャネルに変更
    out_channels=original_conv.out_channels,  # 元の出力チャネル数
    kernel_size=original_conv.kernel_size,  # 元のカーネルサイズ
    stride=original_conv.stride,  # 元のストライド
    padding=original_conv.padding,  # 元のパディング
    bias=original_conv.bias is not None  # 元のバイアス設定
)

# モデルをGPU/CPUに移動
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)



# 5. モデルの学習




# 損失関数と最適化手法を定義
criterion = nn.CrossEntropyLoss()  # クラス分類の損失関数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam最適化

# 学習ループ
epochs = 10
for epoch in range(epochs):
    model.train()  # モデルを訓練モードに設定
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # データをGPUに送る

        optimizer.zero_grad()  # 勾配を初期化
        
        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 損失計算
        
        # 逆伝播
        loss.backward()
        
        # 最適化
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}")

#%%
import torch
import torch.nn.functional as F
import numpy as np

class SatelliteImageTransform:
    def __init__(self, clip_min=0, clip_max=1, resize_size=(224, 224), augmentations=None):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.resize_size = resize_size
        self.augmentations = augmentations

    def __call__(self, image):
        # 1. クリッピング
        image = np.clip(image, self.clip_min, self.clip_max)
        
        # 2. 正規化
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        std = np.std(image, axis=(1, 2), keepdims=True)
        image = (image - mean) / (std + 1e-8)
        
        # 3. Data Augmentation
        if self.augmentations:
            for aug in self.augmentations:
                image = aug(image)
        
        # 4. テンソル変換とリサイズ
        image = torch.tensor(image, dtype=torch.float32).permute(0, 1, 2)  # (C, H, W)
        image = F.interpolate(image.unsqueeze(0), size=self.resize_size, mode="bilinear", align_corners=False).squeeze(0)
        
        return image

# Augmentation関数例
def horizontal_flip(image):
    if np.random.rand() > 0.5:
        return np.flip(image, axis=2)  # 横方向を反転
    return image

def vertical_flip(image):
    if np.random.rand() > 0.5:
        return np.flip(image, axis=1)  # 縦方向を反転
    return image

def rotate_image(image):
    k = np.random.choice([0, 1, 2, 3])  # 90度単位で回転
    return np.rot90(image, k, axes=(1, 2))

# 使用例
transform = SatelliteImageTransform(
    clip_min=0,
    clip_max=1,
    resize_size=(224, 224),
    augmentations=[horizontal_flip, vertical_flip, rotate_image]
)

# データ例 (Numpy配列)
sample_image = np.random.rand(7, 32, 32)  # ダミーデータ
transformed_image = transform(sample_image)
print(transformed_image.shape)  # torch.Size([7, 224, 224])
