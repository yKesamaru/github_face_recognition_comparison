"""
dlib_japanesefacev1_auc_roc_random_sampling.py

Summary:
    このスクリプトは、dlibとJAPANESE FACE V1(JAPANESE_FACE_V1.onnx)を使用して
    顔認証モデルの性能を評価し、ROC曲線とAUCスコアをプロットします。
    具体的には以下の処理を行います。

    - 指定された顔画像ディレクトリ内の画像をランダムにペアリングし、同一人物かどうかのラベルを生成。
    - dlibおよびONNXモデル(JAPANESE FACE V1)を用いて、各ペアの距離または類似度を計算。
    - 各モデルごとにROC曲線(AUC含む)をプロットし、画像ファイルに保存。
    - ROC曲線の描画に必要なデータをCSV形式で保存。
    - 処理時間を計測し、サンプル数やAUC等とともに出力。
"""

import os
import csv
import time
import random
import itertools
from typing import List, Tuple

import dlib
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

# ================================
# ディレクトリ設定
# ================================
# 実行時のカレントディレクトリが「~/FACE01_DEV」想定。
# モデルファイルが置かれているディレクトリ
MODEL_DIR = os.path.join("..", "test", "assets", "models")
# 画像が格納されているディレクトリ
IMAGE_DIR = os.path.join("..", "test", "assets", "若年日本人女性")

# ================================
# モデルの準備
# ================================
# -- dlibのモデルをロード --
shape_predictor_path = os.path.join(MODEL_DIR, "shape_predictor_5_face_landmarks.dat")
face_recognition_model_path = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# HOGベース
# face_detector = dlib.get_frontal_face_detector()  # 顔検出器(HOGベース)
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# CNNベースの顔検出を使う場合は、以下を利用してください(コメントアウトを解除・置き換え):
cnn_face_detector_path = os.path.join(MODEL_DIR, "mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)
# ※ CNNベース使用時は、検出結果が "d.rect" でアクセス可能

# -- JAPANESE FACE V1 のモデルをロード --
onnx_model_path = os.path.join(MODEL_DIR, "JAPANESE_FACE_V1.onnx")
onnx_model = onnx.load(onnx_model_path)
ort_session = ort.InferenceSession(onnx_model_path)
input_name = onnx_model.graph.input[0].name

# ================================
# 画像前処理の定義 (JAPANESE FACE V1用)
# ================================
mean_value = [0.485, 0.456, 0.406]
std_value = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # 画像を224x224にリサイズ
    transforms.ToTensor(),           # テンソル化
    transforms.Normalize(            # 正規化
        mean=mean_value,
        std=std_value
    )
])

# ================================
# ユーティリティ関数
# ================================
def extract_person_name(file_path: str) -> str:
    """
    指定したファイルパスから人物名を抽出して返す関数。
    ファイル名に「人物名_〇〇.png」という形式を想定。

    Args:
        file_path (str): ファイルへのパス。

    Returns:
        str: 抽出した人物名。
    """
    filename = os.path.basename(file_path)  # 例: "Alice_01.png"
    parts = filename.split('_')
    return parts[0] if parts else ""

def get_sampled_image_label_pairs(image_dir: str, sample_size: int) -> List[Tuple[str, str, int]]:
    """
    ランダムにサンプリングした画像ペアを生成し、ラベル(同一人物=1, 別人=0)を付加する関数。

    Args:
        image_dir (str): 画像が格納されたディレクトリへのパス。
        sample_size (int): サンプリングするペアの数。

    Returns:
        List[Tuple[str, str, int]]: サンプリングされた(画像パス1, 画像パス2, ラベル)のリスト。
    """
    # ディレクトリ内の.pngファイルをすべて取得
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

    if not image_paths:
        print(f"[Warning] No PNG files found in {image_dir}")
        return []

    # 全ペアを作成
    all_pairs = list(itertools.combinations(image_paths, 2))

    # ランダムに sample_size 件だけ選択
    sampled_pairs = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    pair_list = []
    for img1, img2 in sampled_pairs:
        person1 = extract_person_name(img1)
        person2 = extract_person_name(img2)
        label = 1 if person1 == person2 else 0
        pair_list.append((img1, img2, label))

    print(f"Total sampled pairs: {len(pair_list)}")
    return pair_list

def get_embedding_dlib(image_path: str) -> np.ndarray:
    """
    画像ファイルからdlibの埋め込みベクトル(128次元)を取得する関数。

    Args:
        image_path (str): 画像パス。

    Returns:
        np.ndarray or None: 128次元の埋め込みベクトル。顔が検出できなかった場合は None を返す。
    """
    image = dlib.load_rgb_image(image_path)
    # HOGベースを使うなら下記をコメントインして置き換え
    # detected_faces = face_detector(image, 1)
    # if len(detected_faces) == 0:
    #     return None

    # CNNベース
    detected_faces = cnn_face_detector(image, 1)
    if len(detected_faces) == 0:
        return None
    d = detected_faces[0].rect

    # shape = shape_predictor(image, detected_faces[0])  # hog利用（CPU）
    shape = shape_predictor(image, d)                  # cnn利用（GPU）
    face_descriptor = face_recognition_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

def get_embedding_jface(image_path: str) -> np.ndarray:
    """
    画像ファイルからJAPANESE FACE V1 (JAPANESE_FACE_V1.onnx)の埋め込みベクトルを取得する関数。

    Args:
        image_path (str): 画像パス。

    Returns:
        np.ndarray: モデルが出力する埋め込みベクトル(Numpy配列)。
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).numpy()  # バッチ次元を作り numpy 配列へ

    # ONNXモデルで推論
    embedding = ort_session.run(None, {input_name: image})[0]
    return embedding

def calculate_distances_for_model(
    pairs: List[Tuple[str, str, int]],
    model_type: str,
) -> Tuple[List[float], List[int]]:
    """
    指定したモデルタイプ(dlib or japanese_face)で画像ペアのスコアを計算する関数。

    Args:
        pairs (List[Tuple[str, str, int]]): (画像パス1, 画像パス2, ラベル)のリスト。
        model_type (str): "dlib" または "japanese_face"。

    Returns:
        Tuple[List[float], List[int]]: 計算されたスコア(distances or similarities)のリストとラベルのリスト。
    """
    scores = []
    labels = []

    for (img1, img2, label) in pairs:
        if model_type == "dlib":
            emb1 = get_embedding_dlib(img1)
            emb2 = get_embedding_dlib(img2)
            if emb1 is None or emb2 is None:
                continue  # 顔が検出できなかった場合はスキップ
            # ユークリッド距離
            dist = np.linalg.norm(emb1 - emb2)
            scores.append(dist)
            labels.append(label)

        elif model_type == "japanese_face":
            emb1 = get_embedding_jface(img1)
            emb2 = get_embedding_jface(img2)
            if emb1 is None or emb2 is None:
                continue
            # 余弦類似度 (NumPyベース)
            emb1_flat = emb1.flatten()
            emb2_flat = emb2.flatten()
            cos_sim = np.dot(emb1_flat, emb2_flat) / (
                np.linalg.norm(emb1_flat) * np.linalg.norm(emb2_flat)
            )
            scores.append(float(cos_sim))
            labels.append(label)

        else:
            print(f"Unknown model type: {model_type}")
            return [], []

    return scores, labels

def save_roc_data_to_csv(
    model_type: str,
    fpr: List[float],
    tpr: List[float],
    thresholds: List[float],
    sample_size: int,
    elapsed_time: float
) -> None:
    """
    ROC曲線に必要なデータをCSVファイルに保存する関数。

    Args:
        model_type (str): モデル種別(dlib, japanese_faceなど)。
        fpr (List[float]): False Positive Rate のリスト。
        tpr (List[float]): True Positive Rate のリスト。
        thresholds (List[float]): スレッショルド値のリスト。
        sample_size (int): サンプリングされたペアの数。
        elapsed_time (float): 処理にかかった時間（秒）。
    """
    # 保存先を ../test ディレクトリに指定
    output_path = os.path.join("..", "test", f"{model_type}_roc_data_{sample_size}.csv")
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["FPR", "TPR", "Threshold", "SampleSize", "ElapsedTime"])
        for row in zip(fpr, tpr, thresholds):
            writer.writerow(list(row) + [sample_size, elapsed_time])

    print(f"Saved ROC data for {model_type} as {output_path}")

def plot_roc_curve(
    model_type: str,
    scores: List[float],
    labels: List[int],
    sample_size: int,
    elapsed_time: float,
    distance_based: bool = True
) -> None:
    """
    指定したモデルに対してROC曲線をプロットし、画像ファイルとして保存する関数。

    Args:
        model_type (str): モデル種別(dlib, japanese_faceなど)。
        scores (List[float]): 距離または類似度のリスト。
        labels (List[int]): 対応するラベル(1 or 0)のリスト。
        sample_size (int): サンプリングされたペアの数。
        elapsed_time (float): 処理にかかった時間（秒）。
        distance_based (bool): スコアが「距離ベース」の場合はTrue、「類似度ベース」の場合はFalse。
    """
    plt.figure(figsize=(10, 10))

    if distance_based:
        # dlibは距離が小さいほど類似度が高い -> スコアを負にしてROCを描画
        scores_for_roc = [-s for s in scores]
    else:
        # japanese_faceは余弦類似度が大きいほど類似度が高い -> スコアはそのまま
        scores_for_roc = scores

    fpr, tpr, thresholds = roc_curve(labels, scores_for_roc, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # ROCデータをCSVに保存
    save_roc_data_to_csv(model_type, fpr, tpr, thresholds, sample_size, elapsed_time)

    plt.plot(fpr, tpr, label=f"{model_type} (AUC = {roc_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"ROC Curve for {model_type}\n"
              f"(Sample Size: {sample_size}, Time: {elapsed_time / 60:.1f} min)",
              fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)

    # 保存先を ../test ディレクトリに指定
    output_path = os.path.join("..", "test", f"{model_type}_plot_{sample_size}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve for {model_type} as {output_path}")

def main():
    """
    dlibおよびJAPANESE FACE V1を使用して顔認証モデルの評価を実施するメイン関数。

    Summary:
        - ランダムサンプリングされた画像ペアを生成。
        - 2種類のモデル(dlib, JAPANESE FACE V1)で評価を実行。
        - 各モデルごとにROC曲線をプロットして保存。
        - ROCデータをCSV形式で保存。
        - 処理時間を出力。
    """
    # 画像を格納しているディレクトリを指定
    image_dir = IMAGE_DIR

    # ランダムサンプリングするペア数
    sample_size = 10000

    print("Sampling image pairs...")
    pairs = get_sampled_image_label_pairs(image_dir, sample_size)
    if not pairs:
        print("No image pairs generated. Please check your image directory or file extension.")
        return

    # 評価するモデルタイプのリスト
    model_types = ["dlib", "japanese_face"]

    for model_type in model_types:
        print(f"Processing model: {model_type}...")
        start_time = time.time()
        scores, labels = calculate_distances_for_model(pairs, model_type)
        elapsed_time = time.time() - start_time

        if not scores:
            print(f"No valid embeddings or scores found for {model_type}. Skipping...")
            continue

        # dlibはdistanceベース, japanese_faceはsimilarityベース
        distance_based = True if model_type == "dlib" else False

        plot_roc_curve(
            model_type=model_type,
            scores=scores,
            labels=labels,
            sample_size=sample_size,
            elapsed_time=elapsed_time,
            distance_based=distance_based
        )

if __name__ == "__main__":
    main()
