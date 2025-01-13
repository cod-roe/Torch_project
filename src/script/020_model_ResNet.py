# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

# =================================================
# ResNet18
# =================================================

# ハイパーパラメータの設定
num_epoch = 25
lr = 0.005
batch_size = 1024
train_ratio = 0.75
weight_decay = 5e-4
momentum = 0.9
save_interval = 1  # 保存する間隔（エポック単位）


# ResNet18の読み込み
model = models.resnet18(pretrained=True)


# 最初の畳み込み層の入力チャンネル数を7に変更
original_conv = model.conv1
model.conv1 = nn.Conv2d(
    in_channels=7,  # 入力チャンネル数を7に変更
    out_channels=original_conv.out_channels,  # 元の出力チャンネル数をそのまま使用
    kernel_size=original_conv.kernel_size,  # 元のカーネルサイズをそのまま使用
    stride=original_conv.stride,  # 元のストライドをそのまま使用
    padding=original_conv.padding,  # 元のパディングをそのまま使用
    bias=original_conv.bias is not None,  # 元のバイアスの設定をそのまま使用
)

# モデルの出力層の再定義
model.fc = nn.Linear(in_features=512, out_features=2, bias=True)  # 2048 res50

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# デバイスに移動

model = model.to(device)


# 最適化アルゴリズムと損失関数の設定
optimizer = optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()


# %%

# その他


# # PreResNet-18をロード
# model = timm.create_model('resnet18', pretrained=True)
# # SE-ResNet-50のロード
# model = timm.create_model('seresnet50', pretrained=True)
# EfficientNet-B0をロード
# model = models.efficientnet_b0(pretrained=True)
# model = models.resnet34(pretrained=True)

# model = models.googlenet(pretrained=True)
# model = models.mobilenet_v2(pretrained=True)
# model = models.mobilenet_v1(pretrained=True)


