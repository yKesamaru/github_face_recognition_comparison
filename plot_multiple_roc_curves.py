import pandas as pd  # CSVファイルの読み込みに使用
import matplotlib.pyplot as plt  # グラフ描画に使用
from sklearn.metrics import auc  # AUC(Area Under the Curve)を計算する関数
import glob  # 指定したパターンに合致するファイルを検索するモジュール
import os  # ファイルパスを扱うためのモジュール

def plot_multiple_roc_curves(folder_path):  # 複数のROC曲線を1つの図にまとめて描画する関数
    """
    この関数は指定したフォルダ内にある
    「_roc_data_10000.csv」という名前を含む全てのCSVファイルを読み込み、
    そこに含まれるFPR(False Positive Rate)とTPR(True Positive Rate)をもとに
    複数のROC曲線を1枚のプロットに描画し、AUC(Area Under the Curve)を計算して凡例に表示します。

    Parameters
    ----------
    folder_path : str
        CSVファイルが格納されているフォルダのパス。
    """  # 関数のドキュメント

    csv_files = glob.glob(os.path.join(folder_path, "*_roc_data_10000.csv"))  # CSVファイルのパス取得

    plt.figure(figsize=(10, 10))

    for csv_file in csv_files:  # 検索したCSVファイルのリストをループ
        df = pd.read_csv(csv_file)  # CSVファイルをDataFrameとして読み込む
        fpr = df["FPR"].values  # データフレームからFPRの値を配列として取得
        tpr = df["TPR"].values  # データフレームからTPRの値を配列として取得

        roc_auc = auc(fpr, tpr)  # AUC(Area Under the Curve)を算出

        label_name = os.path.basename(csv_file).replace("_roc_data_10000.csv", "")  # ファイル名からモデル名を抽出
        plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.4f})")  # ROC曲線の描画

    plt.plot([0, 1], [0, 1], "k--", label="random")  # ベースラインを点線で描画

    plt.xlim([0.0, 1.0])  # x軸(FPR)の表示範囲
    plt.ylim([0.0, 1.0])  # y軸(TPR)の表示範囲
    plt.xlabel("False Positive Rate")  # x軸のラベル
    plt.ylabel("True Positive Rate")  # y軸のラベル
    plt.title("Multiple models ROC")  # タイトルの設定
    plt.legend(loc="lower right")  # 凡例を右下に配置
    plt.grid(True)  # グリッド線の表示

    # plt.show()を削除した理由: グラフを画面に表示するのではなく、ファイルとして保存するため。
    # plt.show()

    plt.savefig("roc_plot.png")  # 描画したグラフをPNG画像として保存する

# メインブロックを追加
if __name__ == "__main__":  # Pythonスクリプトが直接実行された時だけ以下を実行
    """
    テスト用コード:
    同じフォルダ内にある *_roc_data_10000.csv ファイルを検索し、
    1つのプロットにROC曲線を描画してPNG画像で保存する。
    """  # ドキュメント
    plot_multiple_roc_curves(".")  # 関数を呼び出し
