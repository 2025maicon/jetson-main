# perception/lane.py
"""
카메라 프레임에서 흰색 차선을 찾아
- 조향 오차(steering_error)를 계산하고
- (옵션) 포트홀 유무를 대략 판단하는 모듈.

steering_error:
    0.0  → 중앙
    음수 → 왼쪽으로 치우침
    양수 → 오른쪽으로 치우침
"""

import cv2
import numpy as np
from config import HSV_WHITE_LOWER, HSV_WHITE_UPPER, ROI_RATIO


def process_lane(frame):
    """
    frame (numpy.ndarray, BGR): 카메라에서 읽은 한 프레임

    return:
        steering_error (float): -1.0 ~ 1.0 사이의 값 (대략)
        pothole_detected (bool): 포트홀 감지 여부 (지금은 간단한 placeholder)
    """
    h, w, _ = frame.shape

    # 1. BGR -> HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 흰색 영역 마스크
    lower = np.array(HSV_WHITE_LOWER, dtype=np.uint8)
    upper = np.array(HSV_WHITE_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # 3. ROI (이미지의 아래쪽 일부만 사용)
    roi_top = int(h * (1.0 - ROI_RATIO))
    roi = mask[roi_top:, :]

    # 4. ROI에서 흰색 픽셀들의 x좌표 평균 계산
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        # 차선이 안 보이는 경우: 에러 0으로 두고, 상위 로직에서 따로 처리 가능
        steering_error = 0.0
    else:
        lane_center_x = xs.mean()
        image_center_x = w / 2.0
        # -1.0 ~ 1.0 범위로 정규화
        steering_error = (lane_center_x - image_center_x) / (w / 2.0)

    # 5. 포트홀 감지는 일단 placeholder (나중에 YOLO 결과와 합칠 수도 있음)
    # 예시: 화면 중앙 아래쪽에서 "매우 어두운" 픽셀이 많으면 포트홀로 가정
    pothole_detected = False
    # TODO: 필요하면 간단한 어둠 기반 포트홀 감지 구현

    return float(steering_error), pothole_detected
