"""
포트홀 감지 모듈
"""
import cv2
import numpy as np
import time
from utils.config import (
    POTHOLE_DETECT_FRAMES, POTHOLE_COOLDOWN, POTHOLE_MIN_ABS,
    POTHOLE_RATIO, EMA_ALPHA
)

# 전역 상태 변수
pothole_counter = 0
pothole_last_time = 0.0
baseline_white = None  # EMA baseline for white fraction

# morphology kernel
_morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def detect_pothole(frame_bgr):
    """Compute white fraction in ROI with noise filtering.
    Returns: white_frac (0..1), roi_visual, mask_visual
    """
    h, w = frame_bgr.shape[:2]
    roi_y1 = int(h * 0.60)
    roi_y2 = h
    roi_x1 = int(w * 0.10)
    roi_x2 = int(w * 0.90)
    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    S_MAX = 60
    V_MIN = 180
    lower_white = np.array([0, 0, V_MIN], dtype=np.uint8)
    upper_white = np.array([180, S_MAX, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 노이즈 제거: open + small closing
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, _morph_kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, _morph_kernel)

    white_count = int(np.sum(white_mask == 255))
    total = white_mask.size
    white_frac = white_count / float(total) if total > 0 else 0.0

    white_mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    return white_frac, roi, white_mask_bgr


def should_trigger_pothole(white_frac):
    """Decide based on EMA baseline + absolute minimum and ratio."""
    global baseline_white
    if baseline_white is None:
        baseline_white = white_frac
        return False

    # Update baseline conservatively when white_frac is not very low
    # This prevents fast baseline drop during true pothole events
    if white_frac > baseline_white * 0.9:
        baseline_white = EMA_ALPHA * white_frac + (1 - EMA_ALPHA) * baseline_white

    threshold = max(POTHOLE_MIN_ABS, baseline_white * POTHOLE_RATIO)
    return white_frac < threshold


def avoid_pothole_left(tiki):
    """Left avoidance: stop -> short reverse -> gentle left sweep -> stop
    Tweak timings/speeds to your robot.
    """
    global pothole_last_time
    print("[AVOID] 포트홀 회피 시작 (좌회전)")
    try:
        tiki.stop()
        time.sleep(0.12)

        # try short reverse if supported
        try:
            tiki.backward(50)
            time.sleep(0.5)
        except Exception:
            tiki.stop()
            time.sleep(0.2)

        # left sweep: left slower/backward, right forward -> gentle left
        try:
            tiki.counter_clockwise(25)
            time.sleep(1.8)
            tiki.forward(50)
            time.sleep(2)
            tiki.set_motor_power(tiki.MOTOR_LEFT, 60)
            tiki.set_motor_power(tiki.MOTOR_RIGHT, 10)
            time.sleep(1.8)
            tiki.forward(50)
            time.sleep(2)
        except Exception:
            tiki.stop()
            time.sleep(0.3)

        tiki.stop()
    except Exception as e:
        print("[AVOID][ERROR] 회피중 예외:", e)
        try:
            tiki.stop()
        except Exception:
            pass

    pothole_last_time = time.time()
    print("[AVOID] 회피 완료")


def check_and_handle_pothole(bird, tiki):
    """포트홀 감지 및 회피 처리 (전역 상태 관리 포함)"""
    global pothole_counter, pothole_last_time
    
    white_frac, _roi, _mask = detect_pothole(bird)
    trigger = should_trigger_pothole(white_frac)

    if trigger and (time.time() - pothole_last_time) > POTHOLE_COOLDOWN:
        pothole_counter += 1
    else:
        pothole_counter = 0

    if pothole_counter >= POTHOLE_DETECT_FRAMES:
        print(f"[POTHOLE] 감지: white_frac={white_frac:.3f} baseline={baseline_white:.3f} → 회피 수행")
        avoid_pothole_left(tiki)
        pothole_counter = 0

