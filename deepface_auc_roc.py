"""deepface_auc_roc.py.

Summary:
    このスクリプトは、DeepFaceライブラリを使用して複数の顔認証モデルの性能を評価し、
    ROC曲線とAUCスコアをプロットします。特定の顔画像データセットを使用し、異なるモデル間での
    類似性判定精度を比較する目的で設計されています。

    - 指定された顔画像ディレクトリ内の画像をペアリングし、同一人物かどうかのラベルを生成。
    - DeepFaceライブラリを利用して、複数のモデルで顔画像ペア間の距離を計算。
    - モデルごとのROC曲線をプロットし、AUCスコアを表示。

Note:
    - スクリプトは画像名から人物名を抽出し、同一人物ラベルを作成します。ファイル名の命名規則に依存するため、
    必要に応じて `extract_person_name` 関数を修正してください。

License:
    This script is licensed under the terms provided by yKesamaru, the original author.
"""

import itertools
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from sklearn.metrics import auc, roc_curve


def extract_person_name(file_path: str) -> str:
    """
    指定したファイルパスから人物名を抽出して返す関数

    Args:
        file_path (str): ファイルへのパス

    Returns:
        str: 抽出した人物名
    """
    # ファイル名のみを抽出
    filename = os.path.basename(file_path)  # 例: "髙橋ひかる_428g.jpg_default..png.png.png_0.png_0_align_resize.png"

    # アンダースコアでsplitして先頭を人物名として扱う（例: "髙橋ひかる"）
    # 必要に応じて正規表現などで余計な文字を取り除いてもよい
    person_name = filename.split('_')[0]
    parts = filename.split('_')
    if not parts:
        return ""
    person_name = parts[0]
    print(f"Extracted person name: {person_name} from file: {filename}")

    return person_name


def get_image_label_pairs(image_dir: str) -> List[Tuple[str, str, int]]:
    """
    画像ディレクトリから、(画像パス1, 画像パス2, ラベル[同一人物=1/別人=0]) のリストを返す関数

    Args:
        image_dir (str): 画像が格納されたディレクトリへのパス

    Returns:
        List[Tuple[str, str, int]]: (画像パス1, 画像パス2, ラベル)のタプルを格納したリスト
    """
    # ディレクトリ内のpng画像を取得
    image_paths = []
    for f in os.listdir(image_dir):  # ディレクトリ内を走査
        if f.endswith(".png"):  # PNGファイルのみを対象
            image_paths.append(os.path.join(image_dir, f))  # フルパスを作成してリストに追加

    # (画像パス1, 画像パス2, ラベル)を格納するリスト
    pair_list = []

    # すべての画像ペアを取得
    for img1, img2 in itertools.combinations(image_paths, 2):
        person1 = extract_person_name(img1)  # 1枚目の人物名を取得
        person2 = extract_person_name(img2)  # 2枚目の人物名を取得

        # 同一人物ならラベル=1、別人ならラベル=0
        label = 1 if person1 == person2 else 0

        pair_list.append((img1, img2, label))

    print(f"Total image pairs generated: {len(pair_list)}")
    return pair_list


def calculate_distances_for_model(
    pairs: List[Tuple[str, str, int]],
    model_name: str,
    # detector_backend: str = "dlib",
    # distance_metric: str = "cosine",
    enforce_detection: bool = False,
    align: bool = False
) -> Tuple[List[float], List[int]]:
    """
    DeepFaceのverifyメソッドを使って、指定モデルで画像ペアの距離を計算しラベルと共に返す関数

    Args:
        pairs (List[Tuple[str, str, int]]): (画像パス1, 画像パス2, ラベル) のリスト
        model_name (str): DeepFaceで使用するモデル名
        detector_backend (str, optional): 顔検出器バックエンド. Defaults to "dlib".
        distance_metric (str, optional): 類似度の距離指標. Defaults to "cosine".
        enforce_detection (bool, optional): 顔検出失敗時にエラーを投げるかどうか. Defaults to False.
        align (bool, optional): 顔画像のアライメントを行うか. Defaults to False.

    Returns:
        Tuple[List[float], List[int]]:
            - distances: 各ペアの距離（DeepFace.verifyの結果のdistance）
            - labels:  各ペアの同一人物ラベル(1)または別人ラベル(0)
    """
    distances = []  # 距離を格納するリスト
    labels = []     # ラベルを格納するリスト

    for img1, img2, label in pairs:
        try:
            # DeepFace.verifyを用いて画像ペアの距離を計算
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=model_name,
                # detector_backend=detector_backend,
                # distance_metric=distance_metric,
                enforce_detection=enforce_detection,
                align=align
            )
            # 距離を取得
            distance = result["distance"]
            distances.append(distance)
            labels.append(label)
        except Exception as e:
            # 何らかのエラーが出た場合は、ログだけ出して無視する
            print(f"Error comparing {img1} and {img2} with model={model_name}: {e}")
    if not distances:
        print(f"No distances calculated for model {model_name}.")
    return distances, labels


def plot_roc_curves(
    model_results: Dict[str, Tuple[List[float], List[int]]],
    output_path
) -> None:
    """
    複数モデルのROC曲線を1つのグラフにまとめてプロットし、AUCスコアを凡例に表示する関数

    Args:
        model_results (Dict[str, Tuple[List[float], List[int]]]):
            キーがモデル名, 値が (distanceのリスト, labelのリスト) の辞書
    """
    plt.figure(figsize=(8, 6))  # グラフのサイズを設定

    for model_name, (distances, labels) in model_results.items():
        print(f"Model: {model_name}, Number of labels: {len(labels)}")
        print(f"Unique labels: {set(labels)}")
        if not distances or not labels:
            print(f"{model_name} の距離またはラベルが不足しています。スキップします。")
            continue
        # スコアとしては「距離が小さいほど同一人物らしさが高い」ので
        # ROC計算では "1 - distance" や "負の距離" を与えるなど工夫が必要
        # ここでは "distanceをマイナス" にして同一人物でscoreが高くなるようにする
        scores = [-d for d in distances]  # 距離を反転させる
        print(f"Scores for {model_name}: {scores[:5]}")  # スコアの一部を確認

        try:
            fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)  # ROC曲線を計算
            roc_auc = auc(fpr, tpr)  # AUCスコアを計算
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        except Exception as e:
            print(f"Error generating ROC curve for {model_name}: {e}")

    plt.plot([0, 1], [0, 1], "r--", label="Chance")  # ダイアゴナルライン（目安）
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Curves for Different Face Recognition Models", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)

    # グラフを画像ファイルとして保存
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"ROC曲線を {output_path} に保存しました。")


def main():
    """
    全体の処理を行うメイン関数
    - 指定ディレクトリの画像ペアを生成
    - 複数の顔認証モデルで距離を計測
    - ROC曲線とAUCをまとめてプロット
    """
    # 画像が保存されているディレクトリのパス
    # image_dir = "assets/otameshi"
    image_dir = "assets/若年日本人女性"

    # 1. 画像ペアを作成
    print("画像ペアを作成しています...")
    pairs = get_image_label_pairs(image_dir)

    # 評価したいモデルのリスト
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "Dlib",
        "ArcFace",
        "SFace",
        "GhostFaceNet"
    ]

    model_results = {}  # {モデル名: (distances, labels)} の辞書

    for model in models:
        print(f"{model} で距離を計算中...")
        distances, labels = calculate_distances_for_model(
            pairs=pairs,
            model_name=model,
            # detector_backend="dlib",   # detector_backend を dlib に固定
            # distance_metric="cosine",  # デフォルトcosineを利用
            enforce_detection=False,   # 顔検出失敗時にも継続する
            align=False                # 事前にアライメント済みなのでアライメントOFF
        )
        model_results[model] = (distances, labels)

    # ROC曲線をプロット
    print("model_results の内容を確認しています...")
    print(model_results)
    print("ROC曲線をプロットしています...")
    plot_roc_curves(model_results, output_path='plot')


if __name__ == "__main__":
    main()
