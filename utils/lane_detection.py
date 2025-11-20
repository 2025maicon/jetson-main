"""
차선 감지 모듈
"""
import cv2
import numpy as np
from utils.config import BIRD_WIDTH, BIRD_HEIGHT


def sliding_window_center_with_vis(binary_mask, num_windows=8, margin=20, min_pixels=30):
    """슬라이딩 윈도우를 사용하여 차선 중앙 계산 및 시각화"""
    h, w = binary_mask.shape[:2]
    window_height = h // num_windows
    ys, xs = np.where(binary_mask == 255)

    vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    if len(xs) == 0:
        return None, vis

    bottom_thresh = int(h * 0.75)
    bottom_inds = ys >= bottom_thresh
    if np.any(bottom_inds):
        current_x = int(np.mean(xs[bottom_inds]))
    else:
        current_x = int(np.mean(xs))

    centers = []
    pts = []

    for i in range(num_windows):
        y_low = h - (i + 1) * window_height
        y_high = h - i * window_height
        y_low = max(0, y_low)
        if y_high <= y_low:
            continue

        inds = (ys >= y_low) & (ys < y_high)
        win_xs = xs[inds]
        if len(win_xs) == 0:
            continue

        x_min = max(0, current_x - margin)
        x_max = min(w - 1, current_x + margin)

        lane_inds = (win_xs >= x_min) & (win_xs <= x_max)
        if np.sum(lane_inds) < min_pixels:
            continue

        current_x = int(np.mean(win_xs[lane_inds]))
        centers.append(current_x)

        cy = (y_low + y_high) // 2
        pts.append((current_x, cy))

        cv2.rectangle(vis, (x_min, y_low), (x_max, y_high), (0, 255, 0), 1)
        cv2.circle(vis, (current_x, cy), 3, (255, 0, 255), -1)

    if len(centers) == 0:
        return int(np.mean(xs)), vis

    pts_np = np.array(pts, np.int32)
    cv2.polylines(vis, [pts_np], False, (255, 0, 255), 2)

    return int(np.mean(centers)), vis


def detect_lane_center(frame_bgr, white_threshold=180):
    """차선 중앙 감지 및 디버그 정보 반환"""
    h, w = frame_bgr.shape[:2]

    # ROI (하단부)
    roi_y1 = int(h * 0.60)
    roi_y2 = h
    roi_x1 = int(w * 0.10)
    roi_x2 = int(w * 0.90)
    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]

    # ① BGR -> HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 흰색만 남기기 위한 조건
    S_MAX = 60
    V_MIN = 180

    lower_white = np.array([0, 0, V_MIN], dtype=np.uint8)
    upper_white = np.array([180, S_MAX, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 노란색 제거 (필요 시)
    lower_yellow = np.array([15, 80, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))
    white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)

    ys, xs = np.where(white_mask == 255)
    white_mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

    if len(xs) == 0:
        return None, False, {
            "roi": roi,
            "white": white_mask_bgr,
            "sliding": white_mask_bgr.copy(),
        }

    # 슬라이딩윈도우
    center_x_rel, sliding_vis = sliding_window_center_with_vis(white_mask)

    if center_x_rel is None:
        center_x_rel = int(np.mean(xs))

    center_x_frame = roi_x1 + center_x_rel

    # 가로 흰줄 판단 (STOP LINE 등)
    x_range = np.max(xs) - np.min(xs)
    y_range = np.max(ys) - np.min(ys)
    is_horizontal_bar = (x_range > roi_w * 0.6 and y_range < roi_h * 0.25)

    return center_x_frame, is_horizontal_bar, {
        "roi": roi,
        "white": white_mask_bgr,
        "sliding": sliding_vis,
    }


def make_panel(bird, dbg):
    """2×2 디버그 패널 생성"""
    def R(img):
        return cv2.resize(img, (BIRD_WIDTH, BIRD_HEIGHT))

    panel_top = cv2.hconcat([R(bird), R(dbg["roi"])])
    panel_bot = cv2.hconcat([R(dbg["white"]), R(dbg["sliding"])])
    return cv2.vconcat([panel_top, panel_bot])

