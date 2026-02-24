

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import csv
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import cv2 as cv
import numpy as np

# ============================================================
# 0) 固定値（ここだけ触ればOK）
# ============================================================

# --- カメラ ---
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# --- ロボ自己位置（固定） ---
ROBOT_X = 0.0
ROBOT_Y = 0.0
ROBOT_YAW_RAD = 0.0  # [rad] 反時計回り+

# --- ボールの実直径 ---
BALL_DIAM_M = 0.055  # 55mm

# --- カメラ内部パラメータ（暫定）---
FX = 780.0  # [px]
FY = 780.0  # [px]

# --- カメラの機体に対する取り付け（機体座標: x前, y左, z上） ---
CAM_TX = 0.12  # [m]
CAM_TY = 0.00  # [m]
CAM_TZ = 0.25  # [m]

# --- カメラの向き（機体に対して） ---
CAM_YAW = 0.0
CAM_PITCH = 0.0
CAM_ROLL = 0.0

# --- 中心判定 ---
CENTER_TOL_PX = 40

# --- 検出しきい値 ---
AREA_MIN_PX = 400
MIN_CIRCULARITY = 0.70   # 互換用（改良版では主にHULL側使用）
MIN_RADIUS_PX = 20.0

# --- 欠け/横線対策 ---
MORPH_KERNEL_ROUND = (7, 7)    # 通常の欠け埋め
MORPH_KERNEL_H     = (11, 3)   # 横線（帯抜け）埋め
MORPH_KERNEL_V     = (3, 11)   # 縦線対策（必要時のみ）

CLOSE_ITERS_ROUND = 1
CLOSE_ITERS_H     = 1
OPEN_ITERS        = 1
USE_VERTICAL_CLOSE = False     # 縦線が出る環境だけ True

# --- 形判定（欠け許容寄り） ---
MIN_HULL_CIRCULARITY = 0.72
MIN_SOLIDITY         = 0.72
MIN_FILL_RATIO       = 0.45

# --- 背景誤検出対策 ---
EDGE_MARGIN_PX = 8          # 画面端候補を除外（0で無効）
MAX_RADIUS_PX = 180.0       # 巨大候補除外
MIN_BBOX_ASPECT = 0.55      # min(bw,bh)/max(bw,bh)
MIN_EXTENT_CIRCLE = 0.40    # area / (pi*r^2)
CENTER_BIAS_WEIGHT = 0.35   # 中心寄りを優先（0で無効）

# --- ログ ---
CSV_PATH = "ball_world_log.csv"
LOG_COOLDOWN_SEC = 0.25  # 連打防止

# --- 重複除去（フィールド座標） ---
DEDUP_M = 0.10  # 10cm以内なら同じ玉扱い（0で無効）

# --- デバッグ表示 ---
SHOW_MASKS = True
SHOW_PROCESSED_MASKS = True

# ============================================================
# 1) HSV色範囲
# ============================================================

@dataclass
class HSVRange:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]

COLOR_RANGES = {
    "blue":   [HSVRange((90, 50, 50), (140, 255, 255))],
    "yellow": [HSVRange((15, 60, 60), (40, 255, 255))],
    "red":    [HSVRange((0, 120, 70), (10, 255, 255))],
}

COLOR_BGR = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "unknown": (255, 0, 255),
}

# ============================================================
# 2) 小物（FPS / 回転）
# ============================================================

class FPS:
    def __init__(self, avg_over=30):
        self.t = time.time()
        self.deq = deque(maxlen=avg_over)

    def tick(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        if dt > 0:
            self.deq.append(1.0 / dt)

    def get(self):
        return sum(self.deq) / len(self.deq) if self.deq else 0.0

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return ((1,0,0),(0,c,-s),(0,s,c))

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return ((c,0,s),(0,1,0),(-s,0,c))

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return ((c,-s,0),(s,c,0),(0,0,1))

def matmul(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )

def matvec(A, v):
    return (
        A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
        A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
        A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2],
    )

# ============================================================
# 3) 色マスク生成
# ============================================================

def make_mask_for_color(hsv, color_name: str) -> np.ndarray:
    ranges = COLOR_RANGES[color_name]
    mask = None
    for rng in ranges:
        part = cv.inRange(
            hsv,
            np.array(rng.lower, dtype=np.uint8),
            np.array(rng.upper, dtype=np.uint8)
        )
        mask = part if mask is None else cv.bitwise_or(mask, part)
    return mask

# ============================================================
# 4) 検出（欠け/横線/背景対策込み）
# ============================================================

def detect_from_mask(mask: np.ndarray, area_min: int, kernel: np.ndarray,
                     min_circularity: float, min_radius: float,
                     max_count: int = 2) -> List[Dict]:

    if mask is None:
        return []

    H, W = mask.shape[:2]

    # --- カーネル ---
    kernel_round = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_ROUND)
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, MORPH_KERNEL_H)
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, MORPH_KERNEL_V)

    # --- 欠け/横線対策モルフォロジー ---
    proc = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_round, iterations=CLOSE_ITERS_ROUND)
    proc = cv.morphologyEx(proc, cv.MORPH_CLOSE, kernel_h, iterations=CLOSE_ITERS_H)

    if USE_VERTICAL_CLOSE:
        proc = cv.morphologyEx(proc, cv.MORPH_CLOSE, kernel_v, iterations=1)

    proc = cv.morphologyEx(proc, cv.MORPH_OPEN, kernel_round, iterations=OPEN_ITERS)

    # 輪郭をなめらかにする
    proc = cv.GaussianBlur(proc, (5, 5), 0)
    _, mask_bin = cv.threshold(proc, 127, 255, cv.THRESH_BINARY)

    # --- 内部の帯抜け/穴を埋める（外形優先） ---
    contours_ext, _ = cv.findContours(mask_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask_bin)
    if contours_ext:
        cv.drawContours(filled, contours_ext, -1, 255, thickness=cv.FILLED)
        mask_bin = filled

    contours, _ = cv.findContours(mask_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in contours:
        area = cv.contourArea(c)
        if area < area_min:
            continue

        per = cv.arcLength(c, True)
        if per <= 1e-6:
            continue

        # bbox（背景誤検出対策）
        x, y, bw, bh = cv.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue

        # 端に貼り付く候補を除外
        if EDGE_MARGIN_PX > 0:
            if (
                x <= EDGE_MARGIN_PX or
                y <= EDGE_MARGIN_PX or
                (x + bw) >= (W - EDGE_MARGIN_PX) or
                (y + bh) >= (H - EDGE_MARGIN_PX)
            ):
                continue

        # 細長い候補を除外
        aspect = min(bw, bh) / max(bw, bh)
        if aspect < MIN_BBOX_ASPECT:
            continue

        circularity_raw = 4.0 * math.pi * area / (per * per)

        # 凸包（欠け耐性UP）
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        hull_per = cv.arcLength(hull, True)
        if hull_area <= 1e-6 or hull_per <= 1e-6:
            continue

        circularity_hull = 4.0 * math.pi * hull_area / (hull_per * hull_per)
        solidity = area / hull_area

        # 半径は凸包から取ると安定
        (cx0, cy0), radius = cv.minEnclosingCircle(hull)
        if radius < min_radius:
            continue
        if radius > MAX_RADIUS_PX:
            continue

        # モーメント中心
        M = cv.moments(c)
        if M["m00"] <= 1e-6:
            cx, cy = cx0, cy0
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        # 包含円内の色率
        circle_mask = np.zeros_like(mask_bin)
        cv.circle(circle_mask, (int(round(cx0)), int(round(cy0))), int(round(radius)), 255, -1)
        in_circle = cv.bitwise_and(mask_bin, circle_mask)

        circle_px = cv.countNonZero(circle_mask)
        fill_px = cv.countNonZero(in_circle)
        fill_ratio = (fill_px / circle_px) if circle_px > 0 else 0.0

        # 背景対策: 面積 / 外接円面積（スカスカ候補除外）
        extent_circle = (area / (math.pi * radius * radius)) if radius > 1e-6 else 0.0
        if extent_circle < MIN_EXTENT_CIRCLE:
            continue

        # 判定
        if circularity_hull < MIN_HULL_CIRCULARITY:
            continue
        if solidity < MIN_SOLIDITY:
            continue
        if fill_ratio < MIN_FILL_RATIO:
            continue

        # 中心ボーナス（端の誤検出を下げる）
        dx = (cx - (W * 0.5)) / (W * 0.5)
        dy = (cy - (H * 0.5)) / (H * 0.5)
        center_dist_norm = math.sqrt(dx * dx + dy * dy)
        center_bonus = max(0.0, 1.0 - center_dist_norm)

        candidates.append({
            "cx": float(cx),
            "cy": float(cy),
            "radius": float(radius),
            "diam_px": float(radius * 2.0),
            "area": float(area),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "bbox_aspect": float(aspect),
            "circularity": float(circularity_raw),
            "circularity_hull": float(circularity_hull),
            "solidity": float(solidity),
            "fill_ratio": float(fill_ratio),
            "extent_circle": float(extent_circle),
            "center_bonus": float(center_bonus),
            "mask": mask_bin,      # 後処理後マスク
            "mask_raw": mask,      # 生マスク
        })

    def score(d):
        return (
            1.8 * d.get("circularity_hull", 0.0) +
            1.2 * d.get("solidity", 0.0) +
            1.0 * d.get("fill_ratio", 0.0) +
            0.8 * d.get("extent_circle", 0.0) +
            CENTER_BIAS_WEIGHT * d.get("center_bonus", 0.0) +
            0.00035 * d.get("area", 0.0)
        )

    candidates.sort(key=score, reverse=True)
    return candidates[:max_count]

def is_centered(cx: float, cy: float, w: int, h: int, tol_px: int) -> bool:
    return abs(cx - w/2.0) <= tol_px and abs(cy - h/2.0) <= tol_px

# ============================================================
# 5) フィールド座標推定
# ============================================================

def estimate_ball_world_xy(cx_px: float, cy_px: float, diam_px: float,
                           width: int, height: int,
                           robot_pose: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """
    robot_pose: {"x":..., "y":..., "yaw":...} [m,m,rad]
    return: (Xw, Yw) [m]
    """
    if diam_px <= 1.0:
        return None
    if not all(k in robot_pose for k in ("x", "y", "yaw")):
        return None

    CX = width * 0.5
    CY = height * 0.5

    # 見かけ直径 -> 距離Z
    Z = (FX * BALL_DIAM_M) / diam_px

    # OpenCVカメラ座標 (x右,y下,z前)
    Xc = (cx_px - CX) / FX * Z
    Yc = (cy_px - CY) / FY * Z
    Zc = Z
    p_cam = (Xc, Yc, Zc)

    # 機体座標 (x前,y左,z上) に変換
    p_robot0 = (p_cam[2], -p_cam[0], -p_cam[1])

    # 取り付け回転
    R = matmul(rot_z(CAM_YAW), matmul(rot_y(CAM_PITCH), rot_x(CAM_ROLL)))
    p_robot = matvec(R, p_robot0)

    # 取り付け平行移動
    p_robot = (p_robot[0] + CAM_TX, p_robot[1] + CAM_TY, p_robot[2] + CAM_TZ)

    # 機体 -> フィールド (2D)
    rx, ry, ryaw = float(robot_pose["x"]), float(robot_pose["y"]), float(robot_pose["yaw"])
    c, s = math.cos(ryaw), math.sin(ryaw)

    Xw = rx + c * p_robot[0] - s * p_robot[1]
    Yw = ry + s * p_robot[0] + c * p_robot[1]
    return (Xw, Yw)

# ============================================================
# 6) 描画
# ============================================================

def draw_overlay(frame, detections, fps, target_color, target_det, centered):
    h, w = frame.shape[:2]

    # center cross
    cv.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1, cv.LINE_AA)
    cv.line(frame, (0, h // 2), (w, h // 2), (255, 255, 255), 1, cv.LINE_AA)

    # center box
    x1 = int(w / 2 - CENTER_TOL_PX); y1 = int(h / 2 - CENTER_TOL_PX)
    x2 = int(w / 2 + CENTER_TOL_PX); y2 = int(h / 2 + CENTER_TOL_PX)
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_AA)

    # detections
    for cname, dets in detections.items():
        col = COLOR_BGR.get(cname, (255, 255, 255))
        for d in dets:
            cx, cy = int(d["cx"]), int(d["cy"])
            r = int(max(2, d["radius"]))
            cv.circle(frame, (cx, cy), r, col, 2, cv.LINE_AA)
            cv.circle(frame, (cx, cy), 2, col, -1)

            # bbox（背景対策の確認用）
            if "bbox" in d:
                x, y, bw, bh = d["bbox"]
                cv.rectangle(frame, (x, y), (x + bw, y + bh), col, 1, cv.LINE_AA)

            info = (
                f"{cname}:h{d.get('circularity_hull',0):.2f} "
                f"s{d.get('solidity',0):.2f} "
                f"f{d.get('fill_ratio',0):.2f} "
                f"e{d.get('extent_circle',0):.2f}"
            )
            cv.putText(frame, info, (cx + 8, cy - 8), cv.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(frame, info, (cx + 8, cy - 8), cv.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv.LINE_AA)

    txt1 = f"FPS:{fps:4.1f}  TARGET:{(target_color or 'NONE').upper()}  CENTERED:{centered}"
    cv.putText(frame, txt1, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(frame, txt1, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

    if target_det is not None:
        txt2 = (
            f"diam:{target_det['diam_px']:.1f} "
            f"raw:{target_det.get('circularity',0):.3f} "
            f"hull:{target_det.get('circularity_hull',0):.3f}"
        )
        txt3 = (
            f"sol:{target_det.get('solidity',0):.3f} "
            f"fill:{target_det.get('fill_ratio',0):.3f} "
            f"ext:{target_det.get('extent_circle',0):.3f}"
        )
        txt4 = (
            f"area:{target_det.get('area',0):.0f} "
            f"asp:{target_det.get('bbox_aspect',0):.3f} "
            f"ctr:{target_det.get('center_bonus',0):.3f}"
        )

        cv.putText(frame, txt2, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, txt2, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(frame, txt3, (10, 72), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, txt3, (10, 72), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(frame, txt4, (10, 94), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(frame, txt4, (10, 94), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

# ============================================================
# 7) メイン
# ============================================================

def main():
    robot_pose_fixed = {"x": ROBOT_X, "y": ROBOT_Y, "yaw": ROBOT_YAW_RAD}

    # 保存用変数（フィールド座標）
    ball_points_world: List[Tuple[str, int, Tuple[float, float]]] = []
    ball_points_world_by_color: Dict[str, List[Tuple[float, float]]] = {
        "red": [], "blue": [], "yellow": [], "unknown": []
    }
    ball_count_world = {"red": 0, "blue": 0, "yellow": 0, "unknown": 0}
    last_saved_world: Dict[str, Optional[Tuple[float, float]]] = {
        "red": None, "blue": None, "yellow": None, "unknown": None
    }

    # CSV初期化
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow([
            "timestamp",
            "color",
            "world_X_m",
            "world_Y_m",
            "saved_var_name",
            "saved_value_(Y,X)",
        ])

    # カメラ
    cap = cv.VideoCapture(CAM_INDEX, cv.CAP_ANY)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise SystemExit("ERROR: Cannot open camera.")

    fps = FPS()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, MORPH_KERNEL_ROUND)  # 互換引数用
    last_log_t = 0.0

    print("[INFO] qで終了")
    print("[INFO] 中心に入ったボールを world 座標に変換して保存します")
    print("[INFO] 保存変数: blue_1, red_1 ... / ball_points_world / ball_points_world_by_color")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fps.tick()

        h, w = frame.shape[:2]

        # 入力平滑化（HSV範囲は変えない）
        blur = cv.GaussianBlur(frame, (5, 5), 0)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

        detections = OrderedDict()
        masks = {}
        masks_processed = {}

        for cname in ["red", "blue", "yellow"]:
            mask = make_mask_for_color(hsv, cname)
            dets = detect_from_mask(
                mask=mask,
                area_min=AREA_MIN_PX,
                kernel=kernel,              # 互換引数（関数内では独自カーネル使用）
                min_circularity=MIN_CIRCULARITY,
                min_radius=MIN_RADIUS_PX,
                max_count=2
            )
            detections[cname] = dets

            if SHOW_MASKS:
                masks[cname] = mask
            if SHOW_PROCESSED_MASKS:
                if len(dets) > 0 and "mask" in dets[0]:
                    masks_processed[cname] = dets[0]["mask"]
                else:
                    masks_processed[cname] = mask

        # ターゲット選択（最大面積）
        target_color = None
        target_det = None
        best_area = -1.0
        for cname in ["red", "blue", "yellow"]:
            for d in detections[cname]:
                if d["area"] > best_area:
                    best_area = d["area"]
                    target_color = cname
                    target_det = d

        centered = False
        if target_det is not None:
            centered = is_centered(target_det["cx"], target_det["cy"], w, h, CENTER_TOL_PX)

        draw_overlay(frame, detections, fps.get(), target_color, target_det, centered)

        cv.imshow("ball_world_logger", frame)

        if SHOW_MASKS and masks:
            cv.imshow("masks_red_raw", masks["red"])
            cv.imshow("masks_blue_raw", masks["blue"])
            cv.imshow("masks_yellow_raw", masks["yellow"])

        if SHOW_PROCESSED_MASKS and masks_processed:
            cv.imshow("masks_red_proc", masks_processed["red"])
            cv.imshow("masks_blue_proc", masks_processed["blue"])
            cv.imshow("masks_yellow_proc", masks_processed["yellow"])

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # 中心に入ったら保存
        now = time.time()
        if centered and target_det is not None and (now - last_log_t) >= LOG_COOLDOWN_SEC:
            last_log_t = now

            world = estimate_ball_world_xy(
                cx_px=float(target_det["cx"]),
                cy_px=float(target_det["cy"]),
                diam_px=float(target_det["diam_px"]),
                width=w,
                height=h,
                robot_pose=robot_pose_fixed
            )

            if world is None:
                print("[WARN] world coord skipped (check parameters).")
                continue

            Xw, Yw = world
            color_key = target_color if target_color is not None else "unknown"


            yx_world = (Yw, Xw)

            # 重複除去
            do_save = True
            if DEDUP_M > 0 and last_saved_world[color_key] is not None:
                ly, lx = last_saved_world[color_key]
                if abs(Yw - ly) <= DEDUP_M and abs(Xw - lx) <= DEDUP_M:
                    do_save = False

            if not do_save:
                print(f"[SKIP] duplicate near {color_key}: {yx_world}")
                continue

            ball_count_world[color_key] += 1
            var_name = f"{color_key}_{ball_count_world[color_key]}"
            globals()[var_name] = yx_world

            ball_points_world_by_color[color_key].append(yx_world)
            ball_points_world.append((color_key, ball_count_world[color_key], yx_world))
            last_saved_world[color_key] = yx_world

            print(f"[SAVE] {var_name} = (Y,X) {yx_world}")

            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    now,
                    color_key.upper(),
                    f"{Xw:.4f}",
                    f"{Yw:.4f}",
                    var_name,
                    f"{yx_world}",
                ])

    cap.release()
    cv.destroyAllWindows()

    print(f"[INFO] saved CSV: {CSV_PATH}")
    print("[INFO] Variables created:")
    print(" - ball_points_world")
    print(" - ball_points_world_by_color")
    print(" - color_i variables in globals() like blue_1, red_1 ...")

if __name__ == "__main__":

    main()
