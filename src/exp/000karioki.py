

# %% ファイルの読み込み
# Load Data
# =================================================
#  train_1
image_path = "../input/Satellite/train_1/train/train_1.tif"
image = io.imread(image_path)

print(image.shape)

# %%
# 画像データの確認、可視化
fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
    nrows=1, ncols=7, figsize=(10, 3)
)
ax0.imshow(image[:, :, 0])
ax0.set_title("1")
ax0.axis("off")
ax0.set_adjustable("box")

ax1.imshow(image[:, :, 1])
ax1.set_title("B")
ax1.axis("off")
ax1.set_adjustable("box")

ax2.imshow(image[:, :, 2])
ax2.set_title("G")
ax2.axis("off")
ax2.set_adjustable("box")

ax3.imshow(image[:, :, 3])
ax3.set_title("R")
ax3.axis("off")
ax3.set_adjustable("box")

ax4.imshow(image[:, :, 4])
ax4.set_title("5")
ax4.axis("off")
ax4.set_adjustable("box")

ax5.imshow(image[:, :, 5])
ax5.set_title("6")
ax5.axis("off")
ax5.set_adjustable("box")

ax6.imshow(image[:, :, 6])
ax6.set_title("7")
ax6.axis("off")
ax6.set_adjustable("box")

fig.tight_layout()



# %%
# 3.画像データの前処理 正規化
image_rescaled = exposure.rescale_intensity(image)
# %%
# 前処理を行う前
print("最大値：", image.max())
print("最大値：", image.min())

# %%
# 前処理を行った後
print("最大値：", image_rescaled.max())
print("最大値：", image_rescaled.min())


# %%
# モデリング
model = Sequential()

model.add(Conv2D(32, (3, 3), strides=2, activation="relu", input_shape=(32, 32, 7)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), strides=2, activation="relu"))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.summary()
# %%


tqdm.monitor_interval = 0


# %%
def preprocess(image, mode="train"):
    """
    image: shape = (h, w, channel)を想定。
    mode: 'train', 'val', 'test'を想定。
    """
    if mode == "train":
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)

    elif mode == "val":
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)

    elif mode == "test":
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)
    else:
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)

    return image


# %%
def generate_minibatch(data_path, minibatch_meta, mode="train"):
    images = []
    if mode == "train" or mode == "val":
        labels = []
    for data in minibatch_meta.iterrows():
        im_path = os.path.join(data_path, data[1]["file_name"])
        image = io.imread(im_path)

        # preprocess image
        image = preprocess(image, mode=mode)
        image = image.transpose((2, 0, 1))

        if mode == "train" or mode == "val":
            labels.append(data[1]["flag"])

        images.append(image)

    images = np.array(images)
    if mode == "train" or mode == "val":
        labels = np.array(labels)

        return images, labels
    else:
        return images


# %%
def split_data(data, ratio=0.95):
    train_index = np.random.choice(data.index, int(len(data) * ratio), replace=False)
    val_index = list(set(data.index).difference(set(train_index)))
    train = data.iloc[train_index].copy()
    val = data.iloc[val_index].copy()

    return train, val


# %%
def IOU(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p_true_index = np.where(y_true == 1)[0]
    p_pred_index = np.where(y_pred == 1)[0]
    union = set(p_true_index).union(set(p_pred_index))
    intersection = set(p_true_index).intersection(set(p_pred_index))
    if len(union) == 0:
        return 0
    else:
        return len(intersection) / len(union)


# %%


# %%
model.compile(optimizer="adam", loss="softmax cross entropy", metrics=["accuracy"])
# %%

history = model.fit(
    x=x_tr,
    y=y_tr,
    validation_data=(x_va, y_va),
    batch_size=8,
    epochs=20,
    callbacks=[
        ModelCheckpoint(
            filepath="model_keras.weights.h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0,
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.1, patience=5, verbose=1
        ),
    ],
    verbose=1,
)


# %%
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"テストの正解率{test_acc:.2%}")
print(f"テストのloss{test_loss:.2%}")


# %%
# 学習用画像が格納されているディレクトリを指定する
data_path = "../input/Satellite/train_1/train"

# 学習用データを学習用と検証用に改めて分割する
train, val = split_data(data, ratio=0.95)

print("-" * 20, "train", "-" * 20)
print("number of samples:", len(train))
print("number of positives:", train["flag"].sum())
print("nubmer of negatives:", (1 - train["flag"]).sum())
print("-" * 47)

print("-" * 20, "val", "-" * 20)
print("number of samples:", len(val))
print("number of positives:", val["flag"].sum())
print("nubmer of negatives:", (1 - val["flag"]).sum())
print("-" * 45)
# %%

# %%

# %%
# 可視化
param = [["正解率", "accuracy", "val_accuracy"], ["誤差", "loss", "val_loss"]]

plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-")
    plt.plot(history.history[param[i][2]], "o-")
    plt.xlabel("学習回数")
    plt.legend(["訓練", "テスト"], loc="best")
    if i == 0:
        plt.ylim([0, 1])
plt.show()

# %%
# 予測
pre = model.predict(x_test)


plt.figure(figsize=(12, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])

    index = np.argmax(pre[i])
    pct = pre[i][index]
    ans = ""
    if index != y_test[i]:
        ans = "x--o[" + class_names[y_test[i][0]] + "]"
    lbl = f"{class_names[index]}({pct:.0%}){ans}"
    plt.xlabel(lbl)
plt.show()
