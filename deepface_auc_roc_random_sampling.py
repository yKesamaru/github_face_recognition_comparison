"""deepface_auc_roc_random_sampling.py.

Summary:
    このスクリプトは、DeepFaceライブラリを使用して複数の顔認証モデルの性能を評価し、
    ROC曲線とAUCスコアをプロットします。特定の顔画像データセットを使用し、ランダムサンプリングを行いながら、
    異なるモデル間での類似性判定精度を比較する目的で設計されています。

    - 指定された顔画像ディレクトリ内の画像をランダムにペアリングし、同一人物かどうかのラベルを生成。
    - DeepFaceライブラリを利用して、複数のモデルで顔画像ペア間の距離を計算。
    - 各モデルごとにROC曲線をプロットし、AUCスコアと処理時間を表示して画像ファイルに保存。
    - ROC曲線の描画に必要なデータをCSV形式で保存。

Note:
    - スクリプトは画像名から人物名を抽出し、同一人物ラベルを作成します。ファイル名の命名規則に依存するため、
      必要に応じて `extract_person_name` 関数を修正してください。
    - サンプルサイズを調整することで計算負荷をコントロールできます。

License:
    This script is licensed under the terms provided by yKesamaru, the original author.
"""

import itertools
import os
import random
import time
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.metrics import auc, roc_curve

def extract_person_name(file_path: str) -> str:
    """
    指定したファイルパスから人物名を抽出して返す関数。

    Args:
        file_path (str): ファイルへのパス。

    Returns:
        str: 抽出した人物名。

    Example:
        >>> extract_person_name("example_image_001.png")
        "example"

    Note:
        ファイル名の命名規則に依存しているため、必要に応じて修正してください。

    """
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    return parts[0] if parts else ""

def get_sampled_image_label_pairs(image_dir: str, sample_size: int) -> List[Tuple[str, str, int]]:
    """
    ランダムにサンプリングした画像ペアを生成し、ラベルを付加する関数。

    Args:
        image_dir (str): 画像が格納されたディレクトリへのパス。
        sample_size (int): サンプリングするペアの数。

    Returns:
        List[Tuple[str, str, int]]: サンプリングされた(画像パス1, 画像パス2, ラベル)のリスト。

    Example:
        >>> get_sampled_image_label_pairs("images", 100)
        [("img1.png", "img2.png", 1), ...]

    Note:
        サンプリングするペアの数は、全ペア数を超えることはありません。

    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
    all_pairs = list(itertools.combinations(image_paths, 2))

    sampled_pairs = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    pair_list = []
    for img1, img2 in sampled_pairs:
        person1 = extract_person_name(img1)
        person2 = extract_person_name(img2)
        label = 1 if person1 == person2 else 0
        pair_list.append((img1, img2, label))

    print(f"Total sampled pairs: {len(pair_list)}")
    return pair_list

def calculate_distances_for_model(
    pairs: List[Tuple[str, str, int]],
    model_name: str,
    enforce_detection: bool = False,
    align: bool = False
) -> Tuple[List[float], List[int]]:
    """
    指定したDeepFaceモデルを使用して画像ペアの距離を計算する関数。

    Args:
        pairs (List[Tuple[str, str, int]]): 画像ペアとラベルのリスト。
        model_name (str): DeepFaceで使用するモデル名。
        enforce_detection (bool): 顔検出を強制するかどうか。デフォルトはFalse。
        align (bool): 顔画像のアライメントを行うかどうか。デフォルトはFalse。

    Returns:
        Tuple[List[float], List[int]]: 計算された距離と対応するラベルのタプル。

    Example:
        >>> calculate_distances_for_model(pairs, "VGG-Face")
        ([0.5, 0.6, ...], [1, 0, ...])

    Note:
        DeepFaceのモデル名は事前にサポートされている必要があります。

    """
    distances = []
    labels = []

    for img1, img2, label in pairs:
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=model_name,
                enforce_detection=enforce_detection,
                align=align
            )
            distances.append(result["distance"])
            labels.append(label)
        except Exception as e:
            print(f"Error comparing {img1} and {img2} with model={model_name}: {e}")

    return distances, labels

def save_roc_data_to_csv(
    model_name: str,
    fpr: List[float],
    tpr: List[float],
    thresholds: List[float],
    sample_size: int,
    elapsed_time: float
) -> None:
    """
    ROC曲線に必要なデータをCSVファイルに保存する関数。

    Args:
        model_name (str): モデル名。
        fpr (List[float]): False Positive Rate のリスト。
        tpr (List[float]): True Positive Rate のリスト。
        thresholds (List[float]): スレッショルド値のリスト。
        sample_size (int): サンプリングされたペアの数。
        elapsed_time (float): 処理にかかった時間（秒単位）。

    Example:
        >>> save_roc_data_to_csv("VGG-Face", [0.0, 0.1], [0.0, 0.8], [1.0, 0.5], 10000, 678.45)

    Note:
        ファイル名にはモデル名とサンプルサイズが含まれます。

    """
    output_file = f"{model_name}_roc_data_{sample_size}.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["FPR", "TPR", "Threshold", "SampleSize", "ElapsedTime"])
        for row in zip(fpr, tpr, thresholds):
            writer.writerow(list(row) + [sample_size, elapsed_time])

    print(f"Saved ROC data for {model_name} as {output_file}")

def plot_roc_curve(
    model_name: str,
    distances: List[float],
    labels: List[int],
    sample_size: int,
    elapsed_time: float
) -> None:
    """
    特定のモデルに対してROC曲線をプロットし、画像ファイルとして保存する関数。

    Args:
        model_name (str): モデル名。
        distances (List[float]): 計算された距離のリスト。
        labels (List[int]): 対応するラベルのリスト。
        sample_size (int): サンプリングされたペアの数。
        elapsed_time (float): 処理にかかった時間（秒単位）。

    Example:
        >>> plot_roc_curve("VGG-Face", [0.5, 0.6], [1, 0], 100, 678.45)

    Note:
        グラフはモデルごとに個別のファイルに保存されます。

    """
    plt.figure(figsize=(8, 6))

    scores = [-d for d in distances]
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # ROCデータをCSVに保存
    save_roc_data_to_csv(model_name, fpr, tpr, thresholds, sample_size, elapsed_time)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"ROC Curve for {model_name} (Sample Size: {sample_size}, Time: {elapsed_time / 60:.1f} min)", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)

    output_file = f"{model_name}_plot_{sample_size}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve for {model_name} as {output_file}")

def main():
    """
    DeepFaceを使用して顔認証モデルの評価を実施するメイン関数。

    Summary:
        - ランダムサンプリングされた画像ペアを生成。
        - DeepFaceライブラリを使用して複数モデルでの評価を実行。
        - 各モデルごとにROC曲線をプロットして保存。
        - ROCデータをCSV形式で保存。

    Note:
        使用する画像データセットのパスとサンプルサイズを調整可能です。

    """
    image_dir = "assets/若年日本人女性"
    sample_size = 10000

    print("Sampling image pairs...")
    pairs = get_sampled_image_label_pairs(image_dir, sample_size)

    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        # "Dlib",
        "ArcFace",
        "SFace",
        "GhostFaceNet"
    ]

    for model in models:
        print(f"Processing model: {model}...")
        start_time = time.time()
        distances, labels = calculate_distances_for_model(pairs, model)
        elapsed_time = time.time() - start_time
        if distances and labels:
            plot_roc_curve(model, distances, labels, sample_size, elapsed_time)

if __name__ == "__main__":
    main()
