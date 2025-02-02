# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l


from dataclasses import dataclass

# %%
# =================================================
# EfficientNetV2
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


# %%
# クラス重みを設定
# pos_weight = torch.tensor([23.67])  # 正例を負例の23.67倍重視
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# input = torch.randn(3,  requires_grad=True)
# target = torch.empty(3).random_(2)
# print(input, target)
# loss = criterion(input, target)
# print(loss)

# %%
# モデルの定義
# ========================================
"""EfficientNetV2-S 2025/01/09版"""

"""final_in_features
EfficientNetV2-S
1280
EfficientNetV2-M
1280
EfficientNetV2-L
1280

input_size
EfficientNetV2-S	128~300 * 300
EfficientNetV2-M	128~380 * 384
EfficientNetV2-L	128~380 * 480

"""


# カスタム画像分類
class EfficientNetS(nn.Module):
    def __init__(self, num_class, final_in_features=1280, input_channels=6):  #
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
            nn.AdaptiveAvgPool2d((1, 1)),  # サイズ変更に対応
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


# モデルの定義
# カスタム画像分類
"""衛星画像分析で使っているのはこっち
2015/01/14 dataclass付け加える
dataclassを使用することで名前やパラメータの変更するだけで、モデルのクラスはそのまま使える
"""


@dataclass
class ModelConfig:
    model_name: str
    model_fn: callable  # efficientnet_v2_s
    final_in_features: int  # =1280 1408 1280
    num_class: int = 2
    input_channels: int = 6


class EfficientNetV2(nn.Module):
    def __init__(self, config: ModelConfig):  #
        super().__init__()
        # EfficientNetV2-Sの事前学習済みモデルをロード
        base_model = config.model_fn(pretrained=True)

        # 最初の畳み込み層を取得し、6層にカスタマイズ
        original_conv = base_model.features[0][0]  # 最初のConv2d
        base_model.features[0][0] = nn.Conv2d(
            in_channels=config.input_channels,  # 入力チャンネル数を6に変更
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # エンコーダ部分
        self.encoder = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # サイズ調整
            nn.Flatten(1),
        )

        # 分類層
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.final_in_features, config.num_class),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x



#%%
# 使用例
#dataclass設定
efficientnet_v2_s_config = ModelConfig(
    model_name="EfficientNetV2-S",
    model_fn=efficientnet_v2_s,
    final_in_features=1280,
    num_class=2, #出力の数
    input_channels=6,  # カスタム入力チャネル数
)
# EfficientNetV2-Sのモデル
model = EfficientNetV2(efficientnet_v2_s_config)


efficientnet_v2_m_config = ModelConfig(
    model_name="EfficientNetV2-M",
    model_fn=efficientnet_v2_m,
    final_in_features=1280,
    num_class=2, #出力の数
    input_channels=6,  # カスタム入力チャネル数
)
# EfficientNetV2-Mのモデル
model = EfficientNetV2(efficientnet_v2_m_config)


efficientnet_v2_l_config = ModelConfig(
    model_name="EfficientNetV2-L",
    model_fn=efficientnet_v2_l,
    final_in_features=1280,
    num_class=2, #出力の数
    input_channels=6,  # カスタム入力チャネル数
)
# EfficientNetV2-Lのモデル
model = EfficientNetV2(efficientnet_v2_l_config)




# モデルの定義 defバージョン
# =================================================

# タスクに合わせてレイヤを調整
# =================================================


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

    return model, optimizer, criterion


# モデルの定義 関数なしで普通に定義
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
