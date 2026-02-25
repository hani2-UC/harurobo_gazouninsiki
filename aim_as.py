import cv2
import numpy as np
import time
import json
from pathlib import Path


# =========================
# 設定
# =========================
CAMERA_INDEX = 0
JSON_PATH = Path("ball_status.json")
SEND_INTERVAL_SEC = 0.5

# HoughCircles の検出パラメータ（環境に応じて調整）
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 40
HOUGH_PARAM1 = 120   # Canny上位閾値
HOUGH_PARAM2 = 28    # 小さいほど検出しやすい（誤検出増える）
HOUGH_MIN_RADIUS = 8
HOUGH_MAX_RADIUS = 300

# 前処理
BLUR_KERNEL = (9, 9)

# 表示
SHOW_WINDOW = True


def detect_best_circle(frame_bgr):
    """
    画像から最も有力な円を1つ検出して返す。
    戻り値: (cx, cy, r) または None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    # ここでは「一番大きい円」を採用（必要なら中心近いもの優先に変更可）
    best = max(circles, key=lambda c: c[2])
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])

    return (cx, cy, r)


def build_json_payload(frame_w, frame_h, circle):
    """
    必要な情報だけのJSONペイロードを作る
    """
    center_x = frame_w / 2.0

    if circle is None:
        payload = {
            "timestamp": time.time(),
            "detected": False,
            "horizontal_offset_px": None,   # 画面中心基準（右:+ / 左:-）
            "size_radius_px": None,
            "size_diameter_px": None
        }
        return payload

    cx, cy, r = circle
    horizontal_offset_px = float(cx - center_x)

    payload = {
        "timestamp": time.time(),
        "detected": True,
        "horizontal_offset_px": round(horizontal_offset_px, 2),  # 右:+ / 左:-
        "size_radius_px": int(r),
        "size_diameter_px": int(r * 2)
    }
    return payload


def atomic_write_json(path: Path, data: dict):
    """
    JSONを安全に上書き保存（途中破損を避けるため一時ファイル経由）
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした (index={CAMERA_INDEX})")

    last_send_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("フレーム取得失敗")
                break

            frame_h, frame_w = frame.shape[:2]
            circle = detect_best_circle(frame)

            # 0.5秒ごとにJSON上書き
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL_SEC:
                payload = build_json_payload(frame_w, frame_h, circle)
                atomic_write_json(JSON_PATH, payload)
                # 必要ならログ表示
                print(json.dumps(payload, ensure_ascii=False))
                last_send_time = now

            # 確認用表示
            if SHOW_WINDOW:
                vis = frame.copy()

                # 画面中心線
                cx_screen = frame_w // 2
                cv2.line(vis, (cx_screen, 0), (cx_screen, frame_h), (255, 255, 0), 2)

                if circle is not None:
                    cx, cy, r = circle
                    # 円
                    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

                    offset = cx - (frame_w / 2.0)
                    text = f"offset_x={offset:.1f}px  r={r}px"
                    cv2.putText(vis, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "No sphere detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Sphere Detection", vis)

                # q で終了
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()