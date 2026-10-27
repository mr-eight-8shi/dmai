# Dynamic Vector Router & General Models System

A lightweight, modular, and stream-friendly image classification architecture designed to run smoothly even on consumer GPUs with limited VRAM.

Breaking away from conventional approaches that rely on training a single, massive monolithic model or enforcing complex pre-clustering expert setups, this project is built on a practical philosophy: **"Why force a giant model when you can just scatter light, general-purpose models, route data by approximate visual neighborhoods, and let each model learn organically?"**

On the MNIST benchmark, this approach achieves an accuracy of **97.17% on routed classifications**.

---

## 💡 Motivation & Core Philosophy

Building computer vision models in resource-constrained environments often hits common bottlenecks:

* **Resource Limits:** Fine-tuning or retraining massive networks on modest consumer hardware (e.g., RTX 2070 or laptops) is slow and impractical.
* **Continuous Learning Overhead:** Retraining the whole model whenever new classes or datasets are introduced requires high compute and time costs.

> **🌟 The "Just Scatter It" Approach with General-Purpose Models**
> There is no need for strict initial clustering or complex "expert" engineering.
> The system simply routes data into visual neighborhoods (vector spaces) and assigns lightweight CNN models. Even if completely different labels end up in the same neighborhood vector space, **the underlying general-purpose CNN simply ingests the new labels and grows incrementally through local retraining**, naturally adapting to the local label distribution.

This architecture proves that precise prior categorization is unnecessary: simple vector-based routing paired with small, adaptive models delivers both high accuracy and extreme flexibility.

---

## 🏗️ Architecture Overview

```
[Input Image: 28x28]
       ↓ (PixelVectorExtractor / L2-Normalized Raw Pixels)
[Dynamic Vector Router (Cosine Similarity Search)]
       ↓ (Routes to closest neighborhood model)
[SmallLocalExpertCNN (Lightweight General-Purpose Model)]
       ↓ 
[Output: Top-k Ensemble Classification & Uncertainty Filtering]

```

### 1. Feature Extractor (`PixelVectorExtractor`)

Generates normalized feature vectors (e.g., 512-dim) directly from normalized raw pixel values without relying on heavy pre-trained backbones.

### 2. Dynamic Router (`DynamicClusterRouterPipeline`)

* Evaluates streaming data in real time.
* If the cosine similarity with existing vector clusters exceeds a threshold (e.g., `0.35`), the sample is routed to that neighborhood's model.
* If below the threshold, a **new general-purpose model is dynamically spawned on the fly** to handle the new vector region.

### 3. Lightweight General-Purpose Models (`SmallLocalExpertCNN`)

Compact CNN architectures (tens to hundreds of thousands of parameters) assigned to each vector space.

* **Handling Diverse Labels & Individual Growth:** Even when visually similar vectors correspond to completely different labels, periodic local retraining (with early stopping) allows each model to adapt to its specific local data mix and grow independently.
* Trained weights are automatically saved as local `.pth` checkpoints.

---

## 🚀 Benchmark & Performance (MNIST 60k Streaming & 10k Test Evaluation)

### 📊 Streaming Ingestion Behavior

During the 60,000-sample streaming simulation, models (`cluster_001` through `cluster_010`) are created on demand and continuously adapt to their assigned data batches:

```text
📊 [Streaming Progress: 60000/60000]
================================================================================
 cluster_001 Size:[28, 28]     Total: 3000  ACTIVE (acc 98.7%, unfitted:0/3000)  Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_002 Size:[28, 28]     Total: 3000  ACTIVE (acc 98.3%, unfitted:0/3000)  Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_003 Size:[28, 28]     Total: 3000  ACTIVE (acc 99.4%, unfitted:0/3000)  Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_004 Size:[28, 28]     Total: 3000  ACTIVE (acc 99.1%, unfitted:0/3000)  Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_005 Size:[28, 28]     Total: 2905  ACTIVE (acc 0.0%, unfitted:2905/3000) Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 ...
================================================================================
✅ Full-scale streaming ingestion simulation completed.

```

### 🧪 Full Test Set Evaluation (Top-3 Ensemble)

On 10,000 test samples, the router filters out out-of-distribution inputs ("Unknown") while achieving **97.17% accuracy on evaluated predictions**:

```text
===========================================================================
📊 [Full Test Evaluation Results (Top-k Ensemble)]
===========================================================================
 • Total Test Samples    :  10000
 • Correct Predictions   :   8821
 • Unknown / Rejected    :    922 (9.22% of total)
 • Real Overall Accuracy : 88.21%
 • Accuracy (Evaluated)  : 97.17%
---------------------------------------------------------------------------
 🏷️ [Per-Class Breakdown]
 Label |  Total  | Correct | Unknown | Accuracy (Evaluated) | Real Accuracy
---------------------------------------------------------------------------
    0   |     980 |     965 |       6 |               99.08% |        98.47%
    1   |    1135 |    1109 |      18 |               99.28% |        97.71%
    2   |    1032 |     788 |     195 |               94.15% |        76.36%
    3   |    1010 |     976 |      17 |               98.29% |        96.63%
    4   |     982 |     809 |     126 |               94.51% |        82.38%
    5   |     892 |     749 |     121 |               97.15% |        83.97%
    6   |     958 |     648 |     294 |               97.59% |        67.64%
    7   |    1028 |     964 |      31 |               96.69% |        93.77%
    8   |     974 |     887 |      60 |               97.05% |        91.07%
    9   |    1009 |     926 |      54 |               96.96% |        91.77%
===========================================================================

```

---

## 🛠️ Key Advantages

1. **Independent Growth Across Heterogeneous Labels:**
Even if diverse or unexpected labels end up routed into the same vector space, the assigned model trains on its local multi-label batch, growing smarter independently without affecting the rest of the system.
2. **Fast & Effortless Incremental Training:**
Adding new classes or data streams never requires retraining a monolithic network—only the specific local model handling that neighborhood needs a brief, lightweight update.
3. **Friendly to Low-Spec Hardware:**
Runs comfortably on mid-range consumer GPUs (e.g., RTX 2070 or mobile GPUs) with low VRAM footprint.
4. **Built-in Uncertainty Rejection:**
By applying a vector similarity threshold, inputs that fall outside known regions are flagged as "Unknown," preventing low-confidence misclassifications.

[日本語版 Readme]
# Dynamic Vector Router & General Models System

GPUリソースが限られたPCでもサクサク動き、追加学習が容易に行える「動的ベクトルルーター ＆ 汎用一般人モデル（Dynamic Vector Router & General Models System）」のプロトタイプ実装です。

従来の「巨大な単一モデルをすべて学習させるアプローチ」や「最初から完璧なクラスタリング・専門家設計を強いるアプローチ」のジレンマを打破し、「分類なら、無理に巨大なモデルを1つ作るより、モデルを分割して、ご近所のデータを引き受けた『汎用一般人モデル』たちにそれぞれ担当してもらえばいいじゃないか。えイヤーでばらまけばみんないい感じに作業する！」という発想のもと構築されています。MNISTデータセットを用いた検証では、判定成功時精度 **97.17%** を達成しています。

---

## 💡 背景と開発動機：なぜ「モデル分割 ＋ 動的ルーティング」なのか？

AI開発において、次のような壁にぶつかることは少なくありません。

* **リソースの制約:** ハイスペックなGPUがない環境で巨大なモデルをトレーニング・ファインチューニングするのは非現実的。
* **追加学習の難しさ:** 新しいクラスやデータが追加されるたびにモデル全体を再学習（フルスクラッチまたは重いLoRA等）するのはコストがかかる。

> **🌟 「えイヤー」の精神と「汎用一般人モデル」の共生**
> 最初に厳密なクラスタリングや「専門家（エキスパート）」の設計をする必要はありません。
> まずは大まかな「画像の塊（ベクトル空間）」でえイヤーとルーティングし、割り振られた小さなCNNを配置する。新しく違うラベルのデータが同じ塊に流れ込んできても、そのモデルはもともと「汎用モデル」なので、**追加学習を繰り返すことで「この塊にはこういう多様なラベルが混ざるんだな」と各々が自律的に学習し、成長していく**ことができます。

本システムは、「完璧な事前分類ができていなくても、ベクトルベースの動的ルーティングと軽量な汎用モデルの組み合わせで、十分に高精度かつ柔軟に機能する」ことを証明しています。

---

## 🏗️ アーキテクチャ概要

```
[入力画像: 28x28]
       ↓ (PixelVectorExtractor / ピクセル直値ベースの正規化ベクトル化)
[動的ベクトルルーター (Similarity Search)]
       ↓ (類似度判定 & 近くの塊を担当する汎用モデルへルーティング)
[SmallLocalExpertCNN (各塊を担当する「汎用一般人モデル」)]
       ↓ 
[出力: Top-k アンサンブルによる高精度分類 ＆ 不確実性フィルタリング]

```

### 1. 特徴抽出器 (`PixelVectorExtractor`)

複雑な事前学習済みモデルに頼らず、画像をフラット化してL2正規化したピクセル直値ベースの特徴ベクトル（512次元など）を生成。

### 2. 動的ルーター (`DynamicClusterRouterPipeline`)

* ストリーミング状に流れてくるデータをリアルタイムで評価。
* 既存のモデルとの類似度（コサイン類似度）が閾値（例: `0.35`）以上であれば、一番近い「ご近所さん（既存モデル）」へデータをアペンド。
* 閾値未満であれば、新しい塊を担当する「新しい汎用モデル」をその場で自動生成（Dynamically Created）し、即座にルーティング対象へ組み込み。

### 3. 軽量な汎用一般人モデル (`SmallLocalExpertCNN`)

各ベクトル空間の塊を担当する、数万〜数十万パラメータの非常に軽量なCNN構造。

* **異種ラベルの共存と成長:** 類似のベクトルに「全然違うラベル」が混ざってきても、定期的な追加学習（バッチ学習・Early Stopping）によって、各モデルがそれぞれの塊の中での多様なパターンをどんどん学習し、個別に成長していきます。
* データが蓄積されるとローカルストレージへ`.pth`としてモデルキャッシュを保存します。

---

## 🚀 実行結果（MNIST 60,000件ストリーミング学習 ＆ 10,000件全件評価）

### 📊 ストリーミング・動的ルーティングの様子

60,000件のトレーニングデータをストリーミング形式で投入していく中で、必要に応じて自動的に汎用モデル（cluster_001 〜 cluster_010）が立ち上がり、それぞれの塊で柔軟に学習・成長していく様子が確認できます。

```text
📊 [Streaming Progress: 60000/60000]
================================================================================
 cluster_001 サイズ:[28, 28]     累計: 3000枚  ACTIVE (acc 98.7%, unfitted:0/3000)  クラス: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_002 サイズ:[28, 28]     累計: 3000枚  ACTIVE (acc 98.3%, unfitted:0/3000)  クラス: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_003 サイズ:[28, 28]     累計: 3000枚  ACTIVE (acc 99.4%, unfitted:0/3000)  クラス: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_004 サイズ:[28, 28]     累計: 3000枚  ACTIVE (acc 99.1%, unfitted:0/3000)  クラス: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 cluster_005 サイズ:[28, 28]     累計: 2905枚  ACTIVE (acc 0.0%, unfitted:2905/3000) クラス: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 ...
================================================================================
✅ 全件シミュレーションが完了しました。

```

### 🧪 テストデータ全件評価結果（Top-3 アンサンブル版）

10,000件のテストデータを用いた評価では、類似度が閾値に満たない未知サンプルを「分からない（Unknown）」として適切に弾きつつ、判定を行ったサンプルにおいては**97.17%という高精度**を叩き出しています。

```text
===========================================================================
📊 【テストデータ全件評価結果（Top-k アンサンブル版）】
===========================================================================
 • テストデータ総数    :   10000 件
 • 正解数              :    8821 件
 • 「分からない」件数 :     922 件 (全体比: 9.22%)
 • 全体正解率 (総合)   : 88.21%
 • 判定成功時精度      : 97.17%
---------------------------------------------------------------------------
 🏷️ 【ラベル別 詳細内訳】
 Label |  Total  | Correct | Unknown | Accuracy (判定成功時) | Real Accuracy (全体比)
---------------------------------------------------------------------------
    0   |     980 |     965 |       6 |                99.08% |                98.47%
    1   |    1135 |    1109 |      18 |                99.28% |                97.71%
    2   |    1032 |     788 |     195 |                94.15% |                76.36%
    3   |    1010 |     976 |      17 |                98.29% |                96.63%
    4   |     982 |     809 |     126 |                94.51% |                82.38%
    5   |     892 |     749 |     121 |                97.15% |                83.97%
    6   |     958 |     648 |     294 |                97.59% |                67.64%
    7   |    1028 |     964 |      31 |                96.69% |                93.77%
    8   |     974 |     887 |      60 |                97.05% |                91.07%
    9   |    1009 |     926 |      54 |                96.96% |                91.77%
===========================================================================

```

---

## 🛠️ なぜこの設計が強いのか？（メリット）

1. **類似ベクトルへの異種ラベル混入もOK & 各々が成長:**
たとえ似たような画像ベクトルの中に全く違うラベルのデータが混ざってきても大丈夫です。各モデルは「汎用一般人」として追加学習を重ねるため、担当エリアに流れてくる多様なデータを学習して、**個別のモデルが各々で賢く成長**していくことができます。
2. **追加学習が爆速で簡単:**
新しいデータやクラスが来ても、モデル全体を再トレーニングする必要は一切なし。該当する塊を担当するモデルだけに少量のデータを突っ込んでサクッと追加学習させれば完了です。
3. **ローカル・低スペックPCに優しい:**
NVIDIA RTX 2070のようなミドルレンジGPUやVRAMが限られた環境でも、軽量なモデルの集まりなので快適に動作します。
4. **「分からない」を安全に弾ける:**
類似度閾値を設けているため、どのモデルの担当エリアにも属さない外れ値やノイズは「分からない」と安全に弾くことができます。

---
