
#%%
# 前処理クラス
class ImageTransform:
    '''
    画像の前処理クラス
    学習時と検証時で挙動を変える
    外れ値があるためクリッピング→正規化

    Attributes
    --------------------
    min :
    mean : tupple
        各チャネルの平均値
    std : tupple
        各チャネルの標準偏差
    '''
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToTensor(), #テンソル変換
                transforms.RandomHorizontalFlip(), #水平反転(ランダム)
                transforms.RandomVerticalFlip(), #垂直反転(ランダム)
                transforms.RandomAffine([-30, 30]), #回転(ランダム)
                #transforms.Normalize(mean, std) #標準化
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(), #テンソル変換
                #transforms.Normalize(mean, std) #標準化
            ])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
    

#%%
train_dir = INPUT_PATH / "train/"


class SatelliteDataset(Dataset):
    """
    1. __init__:初期化を行う。
    2. __len__:1エポックあたりに使用するデータ数を返す。
    3. __getitem__:データの読み込み、前処理を行った上で、入力画像と正解ラベルのセットを返す。
    """

    def __init__(
        self, dir, file_list, min, max, transform=None, phase="train", channel="RGB" ):
        """
        衛星画像の学習用データセット

        Attributes
        -------------------
        dir : str
            画像が保管されているパス
        file_list : dataframe
            画像のファイル名とフラグが格納されているデータフレーム
        min : int
            画素値の最小値(クリッピング用)
        max : int
            画素値の最大値(クリッピング用)
        transform : torchvision.transforms.Compose
            前処理パイプライン
        phase : str
            学習か検証かを選択
        channel : str
            使用するチャネル(初期値はRGB)
        """
        self.dir = dir
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.image_path = file_list["file_name"].to_list()
        self.image_label = file_list["flag"].to_list()
        self.channel = channel
        self.min = min
        self.max = max

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # 画像をロード
        img_path = self.image_path[idx]
        img = io.imread(self.dir / img_path)
        img = np.clip(img, self.min, self.max)
        img = (img - self.min) / (self.max - self.min)
        # RGB指定があれば次元を限定する
        if self.channel == "RGB":
            img = img[:, :, ::-1]  # BGR -> RGB
        else:
            pass
        # 前処理の実装
        if self.transform:
            img = self.transform(img, self.phase)
        label = self.image_label[idx]

        return img, label