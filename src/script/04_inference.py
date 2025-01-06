# %%
# 検証フェーズ
# DataLoaderを使って検証

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


def predict_eval(model=model, eval_loader=eval_loader):
    eval_loss = 0.0
    pred_list = []
    true_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(eval_loader):
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

    epoch_loss = eval_loss / len(eval_loader.dataset)
    _, fp, fn, tp = confusion_matrix(true_list, pred_list).flatten()
    eval_iou = tp / (tp + fp + fn)
    print(f"検証データIoU: {eval_iou:.4f}")

    return epoch_loss, eval_iou


model, optimizer, criterion = create_model_efiv2s(
    lr=lr,
    weight_decay=weight_decay,
)
# %%
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

            pred_list.extend(preds.to("cpu").numpy())

    print("Done!")
    return pred_list
