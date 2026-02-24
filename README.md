# Ball World Logger（色ボール検出・ワールド座標保存）

カメラ映像から **赤 / 青 / 黄** のボールを検出し、画面中央に入ったボールの座標を **ワールド座標（X, Y）** に変換して保存する Python スクリプトです。

この版は以下の対策を入れています。

- 欠けた丸に強い（`convexHull`, `solidity`, `fill_ratio`）
- 横線で分断された丸に強い（横長カーネルで `CLOSE`）
- 背景の誤検出を減らす（端除外 / bbox比 / extent / 半径上限 / 中心優先）

---

## できること

- HSVしきい値で **赤 / 青 / 黄** の領域を抽出
- 「丸らしさ」を複数指標で判定してボール候補を検出
- 画面中央に入ったボールを保存（連打防止あり）
- ボールの画面上の見かけ直径から距離を推定（単眼サイズ法）
- カメラ取り付け位置・角度を考慮してワールド座標へ変換
- CSV保存
- デバッグ用に **生マスク / 処理後マスク** を表示

---

## 動作環境

- Python 3.9+（目安）
- OpenCV (`opencv-python`)
- NumPy

---

## インストール

```bash
pip install opencv-python numpy
```

---

## 実行方法

```bash
python タイトル無し3.py
```

> ファイル名は任意です。保存した `.py` の名前で実行してください。

---

## 終了方法

- `q` キーで終了

---

## 保存の動作

プログラムは毎フレーム検出を行い、**最も大きい候補** をターゲットとして扱います。ターゲットが画面中央（中心ボックス内）に入ると、ワールド座標を保存します。

### 保存される内容
- コンソール表示（例: `blue_1 = (Y,X) (...)`）
- `globals()` に変数作成（例: `blue_1`, `red_2`）
- `ball_points_world`
- `ball_points_world_by_color`
- CSVファイル（既定: `ball_world_log.csv`）

---

## 出力CSV形式

`CSV_PATH = "ball_world_log.csv"` に保存されます。


---

## 画面表示（デバッグ）

### メインウィンドウ
- 検出円
- 中心十字
- 中央判定ボックス
- 候補の bbox（矩形）
- HUD（FPS / 指標値）

### マスクウィンドウ（有効時）
- `masks_red_raw` / `masks_blue_raw` / `masks_yellow_raw`（HSVで切った生マスク）
- `masks_red_proc` / `masks_blue_proc` / `masks_yellow_proc`（処理済みマスク）

---

## パラメータ一覧（よく触るところ）

コード冒頭の「固定値」セクションで調整できます。

### 1. カメラ設定

```python
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
```

### 2. ロボット姿勢（固定）

```python
ROBOT_X = 0.0
ROBOT_Y = 0.0
ROBOT_YAW_RAD = 0.0
```

### 3. カメラ・ボールの幾何パラメータ

```python
BALL_DIAM_M = 0.055
FX = 780.0
FY = 780.0
```

### カメラ取り付け位置・姿勢

```python
CAM_TX = 0.12
CAM_TY = 0.00
CAM_TZ = 0.25

CAM_YAW = 0.0
CAM_PITCH = 0.0
CAM_ROLL = 0.0
```

### 4. 中心判定

```python
CENTER_TOL_PX = 40
```

### 5. 色設定（HSV）

```python
COLOR_RANGES = {
    "blue":   [HSVRange((90, 50, 50), (140, 255, 255))],
    "yellow": [HSVRange((15, 60, 60), (40, 255, 255))],
    "red":    [HSVRange((0, 120, 70), (10, 255, 255))],
}
```

### 6. 欠け・横線対策（重要）

```python
MORPH_KERNEL_ROUND = (7, 7)
MORPH_KERNEL_H     = (11, 3)
MORPH_KERNEL_V     = (3, 11)

CLOSE_ITERS_ROUND = 1
CLOSE_ITERS_H     = 1
OPEN_ITERS        = 1
USE_VERTICAL_CLOSE = False
```

### 7. 形判定（欠け許容）

```python
MIN_HULL_CIRCULARITY = 0.72
MIN_SOLIDITY         = 0.72
MIN_FILL_RATIO       = 0.45
```

### 8. 背景誤検出対策（重要）

```python
EDGE_MARGIN_PX = 8
MAX_RADIUS_PX = 180.0
MIN_BBOX_ASPECT = 0.55
MIN_EXTENT_CIRCLE = 0.40
CENTER_BIAS_WEIGHT = 0.35
```

### 背景をまだ拾う場合（おすすめ順）
1. `MIN_EXTENT_CIRCLE = 0.40 -> 0.48`
2. `MIN_SOLIDITY = 0.72 -> 0.78`
3. `EDGE_MARGIN_PX = 8 -> 16`
4. `MIN_BBOX_ASPECT = 0.55 -> 0.65`

### 9. 保存制御・重複除去

```python
LOG_COOLDOWN_SEC = 0.25
DEDUP_M = 0.10
```

---

## こんなことやってるよ

1. BGR画像を軽くぼかす（安定化）
2. HSV変換
3. 色マスク生成（赤 / 青 / 黄）
4. モルフォロジー処理（丸欠け埋め / 横線埋め / ノイズ除去）
5. 輪郭抽出 → 塗りつぶし（内部の帯抜けを埋める）
6. 形判定（凸包円形度 / solidity / fill_ratio / extent_circle / bbox比）
7. スコアリングして上位候補を採用
8. 最大面積の候補をターゲット化
9. 中央に入ったら座標保存

---

## 既知の注意点（重要）

### 1) 距離推定は「単眼サイズ法」
見かけ直径から距離を推定するため、`BALL_DIAM_M` と `FX/FY` の精度に依存します。

### 2) RealSenseの深度は使っていない
この版は通常カメラ入力（単眼）前提です。深度版にすると距離精度を上げられます。

### 3) HSVは環境依存
照明・背景・露出で変わります。必要に応じてHSVの微調整も行ってください。

---

## トラブルシュート

### Q. 背景をまだ拾う
- `MIN_EXTENT_CIRCLE` を上げる
- `MIN_SOLIDITY` を上げる
- `EDGE_MARGIN_PX` を上げる
- `MORPH_KERNEL_H` を大きくしすぎて背景がつながっていないか確認
- 生マスクと処理後マスクを見比べる

### Q. 本物のボールを見失う
- `MIN_FILL_RATIO` を下げる
- `MIN_SOLIDITY` を下げる
- `AREA_MIN_PX` / `MIN_RADIUS_PX` を下げる
- カーネルサイズを小さくする

### Q. 横線（帯抜け）が残る
- `MORPH_KERNEL_H = (11,3) -> (15,3)`
- `CLOSE_ITERS_H = 1 -> 2`
- `SHOW_PROCESSED_MASKS=True` にして処理後マスクを確認

### Q. 保存が連打される / されない
- `LOG_COOLDOWN_SEC` を調整
- `DEDUP_M` を調整
- `CENTER_TOL_PX` を見直す

---



## それでもよくわからない！！

discordでhaniwa_3730まで


