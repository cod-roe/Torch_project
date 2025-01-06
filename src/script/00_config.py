######################
# set dirs & filename
######################
comp_name = "Satellite"

if 'google.colab' in sys.modules:  # colab環境
    print("google.colab")
    INPUT_PATH = Path("/content")  # 読み込みファイル場所
    # name_notebook = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'] # ノートブック名を取得
    name_notebook = "exp_EfficientNetV2-S_1225.ipynb"
    DRIVE = f"/content/drive/MyDrive/Python/SIGNATE/{comp_name}"  # このファイルの親(scr)
 

elif 'kaggle_web_client' in sys.modules:  # kaggle環境
    INPUT_PATH = Path("../input/")

elif 'VSCODE_CWD' in os.environ: # vscode（ローカル）用
    print("vscode")
    INPUT_PATH =  Path(f"../input/{comp_name}")  # 読み込みファイル場所
    abs_path = os.path.abspath(__file__)  # /tmp/work/src/exp/_.py'
    name_notebook = os.path.basename(abs_path) # ノート名を取得
    DRIVE = os.path.dirname(os.getcwd())  # このファイルの親(scr)


#共通
name = os.path.splitext(name_notebook)[0] # 拡張子を除去　filename
OUTPUT = os.path.join(DRIVE, "output")
OUTPUT_EXP = os.path.join(OUTPUT, name)  # logなど情報保存場所
EXP_MODEL = Path(OUTPUT_EXP, "model")  # 学習済みモデル保存

train_dir = INPUT_PATH / "train/" #学習ファイルのフォルダ
pred_sub_dir = INPUT_PATH / "test/" #testファイルのフォルダ

#%%
# %%
# Utilities #
# =================================================
#seedの固定
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)                     # Python標準のランダムシード
    np.random.seed(seed)                  # NumPyのランダムシード
    torch.manual_seed(seed)               # PyTorchのランダムシード（CPU用）
    torch.cuda.manual_seed(seed)          # PyTorchのランダムシード（GPU用）
    torch.cuda.manual_seed_all(seed)      # PyTorchのランダムシード（マルチGPU用）
    torch.backends.cudnn.deterministic = True  # 再現性のための設定
    torch.backends.cudnn.benchmark = False     # 再現性のための設定

seed_everything(seed=123)

# 今の日時
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now


# stdout と stderr をリダイレクトするクラス
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# make dirs
# =================================================
def make_dirs():
    for d in [EXP_MODEL]:
        os.makedirs(d, exist_ok=True)
    print("フォルダ作成完了")

# %%
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

#!%matplotlib inline


# フォルダの作成
make_dirs()
# ファイルの確認
file_list = file_list(INPUT_PATH)

# utils
# ログファイルの設定
logging.basicConfig(
    filename=f"{OUTPUT_EXP}/log_{name}.txt", level=logging.INFO, format="%(message)s"
)
# ロガーの作成
logger = logging.getLogger()

