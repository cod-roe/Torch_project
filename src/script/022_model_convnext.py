# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from torchvision.models import convnext_base, convnext_large

from dataclasses import dataclass
# %%
# =================================================
# ConvNeXt
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
"""ConvNeXt-Base 2025/01/14版"""

"""final_in_features
ConvNeXt-Tiny
768
ConvNeXt-Small
768
ConvNeXt-Base
1024
ConvNeXt-Large
1536
ConvNeXt-XLarge
2048

input_size
convNeXt ALL 224 * 224
"""

# ConvNeXtBase
# ========================================


@dataclass
class ModelConfig:
    model_name: str
    model_fn: callable  # convnext_base convnext_large
    final_in_features: int  # =1024 1536
    num_class: int = 2
    input_channels: int = 6


class ConvNeXt(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # ConvNeXt-Baseの事前学習済みモデルをロード
        base_model = config.model_fn(pretrained=True)

        # 最初の畳み込み層を取得し、カスタマイズ
        original_conv = base_model.features[0][0]  # ConvNeXtの最初のConv2d
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


# %%
# 使用例




convnext_base_config = ModelConfig(
    model_name="ConvNeXt-Base",
    model_fn=convnext_base,
    final_in_features=1024,
    num_class=2,
    input_channels=6,  # カスタム入力チャネル数
)

convnext_large_config = ModelConfig(
    model_name="ConvNeXt-Large",
    model_fn=convnext_large,
    final_in_features=1536,
    num_class=2,
    input_channels=6,
)

# ConvNeXt-Baseのモデル
model = ConvNeXt(convnext_base_config)

# ConvNeXt-Largeのモデル
model = ConvNeXt(convnext_large_config)
