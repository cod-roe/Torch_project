
#%%
"""不均衡データに対して重みをつける
サンプル数を増やすか損失関数に重みをつける
これはオーバーサンプリング
"""
#WeightedRandomSampler 損失関数かどちらか
# # torch を使った方法

# # train_master['flag'] からラベルを取得
# labels =  torch.tensor(train_data["flag"].values)

# # 正例と負例の数を取得
# num_samples = len(labels)
# num_positives = torch.sum(labels == 1).item()
# num_negatives = torch.sum(labels == 0).item()

# print(f'学習データ件数: {num_samples}')
# print(f'正例件数: {num_positives}')
# print(f'負例件数: {num_negatives}')
# print(f'正例割合: {round(num_positives/num_samples, 3)}')


# # DataLoaderのsamplerの重み付きランダムサンプリング（WeightedRandomSampler）
# # 重み計算を正例と負例の実際の数に基づいて行う

# # 重みの計算 (正例: desired_ratio, 負例: 1 - desired_ratio)
# desired_ratio = 0.2 # 例: 正例を20%に増やす

# # 正例と負例に対する重みを計算
# weights = torch.where(
#     labels == 1,  # 正例に対する重み
#     desired_ratio / num_positives,
#     (1 - desired_ratio) / num_negatives,
# )

# weights = weights / weights.sum()

# # WeightedRandomSamplerを設定
# sampler = WeightedRandomSampler(
#     weights, num_samples=num_samples, replacement=True
# )
