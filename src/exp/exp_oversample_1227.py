from sklearn.utils import resample

# 少数クラスと多数クラスを分離
minority_class = data[data["label"] == 1]
majority_class = data[data["label"] == 0]

# 少数クラスを複製
minority_oversampled = resample(
    minority_class,
    replace=True,  # 複製を許可
    n_samples=len(majority_class),  # 多数クラスと同じ数にする
    random_state=42,
)

# 再結合
balanced_data = pd.concat([majority_class, minority_oversampled])


# %%
# 事前に比率を設定
def sampling(df, positive_ratio, over_sampled=True, label_column="flag"):
    """
    サンプリングを実施する関数

    Parameters
    --------------------------
    df : dataframe
        対象データ
    positive_ratio : float
        サンプリング後の正例データの割合
    over_sampled : boolen
        オーバーサンプリングをする場合はTrue
        アンダーサンプリングをする場合はFalse
    label_column : str
        ラベルが格納されているカラム名

    Returns
    --------------------------
    df_sampled : dataframe
        サンプリング後のデータ
    """
    positive_df = df.loc[df[label_column] == 1]
    negative_df = df.loc[df[label_column] == 0]
    if over_sampled:
        size = int(len(negative_df) * (positive_ratio / (1 - positive_ratio)))
        positive_df_sampled = positive_df.sample(n=size, replace=True)
        df_sampled = pd.concat([positive_df_sampled, negative_df], axis=0).reset_index(
            drop=True
        )
    else:
        size = int(len(positive_df) * ((1 - positive_ratio) / positive_ratio))
        negative_df_sampled = negative_df.sample(n=size, replace=False)
        df_sampled = pd.concat([positive_df, negative_df_sampled], axis=0).reset_index(
            drop=True
        )
    return df_sampled


# %%
# DataLoaderのsamplerの重み付きランダムサンプリング（WeightedRandomSampler）
# それの基本
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import torch

# ダミーデータ作成
data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
labels = torch.tensor([0, 0, 1, 1])  # クラスラベル

# サンプルの重み計算 (少数クラスに高い重み)
class_counts = torch.bincount(labels)
weights = 1.0 / class_counts[labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# DataLoaderで使用
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)

# バッチを確認
for batch in dataloader:
    print(batch)

# %%
# 比率を変くできるように
# 正例20%、負例80%の重み計算
desired_ratio = 0.2  # 正例の割合
class_weights = torch.tensor([1 - desired_ratio, desired_ratio])  # 負例, 正例の重み
weights = class_weights[labels]  # ラベルに基づく重み

# WeightedRandomSamplerを設定
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# %%
# より正確
desired_ratio = 0.2  # 例: 正例を20%に増やす
weights = torch.where(
    labels == 1,
    1 / (desired_ratio * len(labels)),  # 正例の重み
    1 / ((1 - desired_ratio) * len(labels)),  # 負例の重み
)
sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

# %%
# 実際のデータに適用
# 重み計算を正例と負例の実際の数に基づいて行う

# torch を使った方法

# train_master['flag'] からラベルを取得
labels =  torch.tensor(train_master["flag"].values)

# 正例と負例の数を取得
num_samples = len(labels)
num_positives = torch.sum(labels == 1).item()
num_negatives = torch.sum(labels == 0).item()

print(f'学習データ件数: {num_samples}')
print(f'正例件数: {num_positives}')
print(f'負例件数: {num_negatives}')
print(f'正例割合: {round(num_positives/num_samples, 3)}')

# DataLoaderのsamplerの重み付きランダムサンプリング（WeightedRandomSampler）
# 重み計算を正例と負例の実際の数に基づいて行う

# 重みの計算 (正例: desired_ratio, 負例: 1 - desired_ratio)
desired_ratio = 0.5 # 例: 正例を50%に増やす

# 正例と負例に対する重みを計算
weights = torch.where(
    labels == 1,  # 正例に対する重み
    (1 - desired_ratio) * num_negatives / num_positives,  # 正例の重み
    desired_ratio * num_positives / num_negatives,  # 負例の重み
)

# WeightedRandomSamplerを設定
sampler = WeightedRandomSampler(
    weights, num_samples=num_samples, replacement=True
)

# DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    sampler=sampler,
)

# %%
# 確認方法
