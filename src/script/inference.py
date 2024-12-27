#%%
# 検証フェーズ
# DataLoaderを使って検証

evalloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
)


eval_loss = 0.0
pred_list = []
true_list = []
model.eval()
with torch.no_grad():
    for images, labels in tqdm(evalloader):
        images = images.float().to(device)
        labels = labels.to(device)

        outputs = model(images)
        # 損失計算
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        eval_loss += loss.item() * images.size(0)
        preds = preds.to("cpu").numpy()
        pred_list.extend(preds)
        labels = labels.to("cpu").numpy()
        true_list.extend(labels)

epoch_loss = eval_loss / len(evalloader.dataset)
tn, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
eval_iou = tp / (tp + fp + fn)
print(eval_iou)


# 8. IoU算出
eval_df = pd.DataFrame(data=[[f, pred, label] for f, pred, label in zip(eval_files.values, prediction, eval_labels.values)],
                       columns=['file_name', 'prediction', 'label'])

tp = len(eval_df.loc[(eval_df['prediction']==1) & (eval_df['label']==1)])
fp = len(eval_df.loc[(eval_df['prediction']==1) & (eval_df['label']==0)])
fn = len(eval_df.loc[(eval_df['prediction']==0) & (eval_df['label']==1)])
iou = tp / (tp + fp + fn)
tn2, fp2, fn2, tp2 = confusion_matrix(eval_df['label'], eval_df['prediction']).flatten()
iou2 = tp2 / (tp2 + fp2 + fn2)
print(iou)
print(f"confmat_ver{iou2}")



def predict_sub(model=model, pred_sub_loader=pred_sub_loader):
    pred_list = []

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(pred_sub_loader):
            images = images.float().to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            preds = preds.to("cpu").numpy()
            pred_list.extend(preds)

    print("Done!")
    return pred_list

#%%
# 推論フェーズ
# DataLoaderを使って推論
pred_sub_dir = INPUT_PATH / "test/"

pred_sub_data = sample_submit.reset_index(drop=True)

pred_sub_dataset = SatelliteDataset(
    dir=pred_sub_dir,
    file_list=pred_sub_data,
    min=3600,
    max=23500,
    transform=ImageTransform(),
    phase="val",
    channel="ALL",
)

pred_sub_loader = DataLoader(
    pred_sub_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


def predict_sub(model=model, pred_sub_loader=pred_sub_loader):
    pred_list = []

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(pred_sub_loader):
            images = images.float().to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            preds = preds.to("cpu").numpy()
            pred_list.extend(preds)

    print("Done!")
    return pred_list
