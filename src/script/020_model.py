# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

# %%
# =================================================
# EfficientNetV2-S
# =================================================

######################
# ハイパーパラメータの設定
num_workers = 2  # DataLoader CPU使用量
epochs = 25
lr = 0.001  # Adam  0.001　SGD 0.005
batch_size = 256
train_ratio = 0.75
weight_decay = 5e-4
# momentum = 0.9
save_interval = 1  # 保存する間隔（エポック単位）


# モデルの定義
# =================================================

# import timm　#modelsで読み込めないときはこっち
# EfficientNetV2-Sをロード
# model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
model = models.efficientnet_v2_s(pretrained=True)


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
model.classifier[1] = nn.Linear(
    in_features=1280, out_features=2, bias=True
)  # in_features使用するモデルによって変更
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, out_features=2, bias=True)　こっちでもいい？

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# デバイスに移動
model = model.to(device)


# 最適化アルゴリズムと損失関数の設定
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


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


# モデルの定義
# =================================================


# タスクに合わせてレイヤを調整
# =================================================
######################
# ハイパーパラメータの設定
num_workers = 2  # DataLoader CPU使用量
epochs = 25
lr = 0.001  # Adam  0.001　SGD 0.005
batch_size = 256
train_ratio = 0.75
weight_decay = 5e-4
# momentum = 0.9
save_interval = 1  # 保存する間隔（エポック単位）


def create_model_efiv2s(
    lr=lr,
    weight_decay=weight_decay,
):
    # =================================================
    # EfficientNetV2-S
    # =================================================

    # import timm #modelsで読み込めないときはこっち
    # EfficientNetV2-Sをロード
    # model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
    model = models.efficientnet_v2_s(pretrained=True)

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
    model.classifier[1] = nn.Linear(
        in_features=1280, out_features=2, bias=True
    )  # in_features使用するモデルによって変更
    # model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, out_features=2, bias=True)　こっちでもいい？

    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # デバイスに移動
    model = model.to(device)

    # 最適化アルゴリズムと損失関数の設定
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    return model,optimizer ,criterion
#%%
# クラス重みを設定
pos_weight = torch.tensor([23.67])  # 正例を負例の23.67倍重視
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
input = torch.randn(3,  requires_grad=True)
target = torch.empty(3).random_(2)
print(input, target)
loss = criterion(input, target)
print(loss)
# %%
#========================================
#2025/01/09版
# モデルの定義
# カスタム画像分類
class EfficientNetS(nn.Module):
    def __init__(self, num_class, final_in_features=1280, input_channels=6):#
        super(EfficientNetS, self).__init__()
        # EfficientNetV2-Sの事前学習済みモデルをロード
        self.model = efficientnet_v2_s(pretrained=True)

        # 最初の畳み込み層を取得し、6層にカスタマイズ
        original_conv = self.model.features[0][0]  # 最初のConv2d
        self.model.features[0][0] = nn.Conv2d(
            in_channels=input_channels,  # 入力チャンネル数を7に変更
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # エンコーダ部分
        self.encoder = nn.Sequential(
            self.model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

        # 分類層
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_in_features, num_class),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


