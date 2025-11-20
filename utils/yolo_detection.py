import cv2
import numpy as np
from collections import defaultdict
from utils.config import OBJECT_NAMES


class YOLODetector:
    """YOLO + ROI + IoU+Distance 기반 신규 객체 판별"""

    def __init__(self, model, aruco_detector=None, conf=0.50):
        self.model = model
        self.aruco_detector = aruco_detector
        self.object_names = OBJECT_NAMES
        self.conf = conf

        # ROI 설정
        self.ROI_BOTTOM_MARGIN = 80    # 아래에서 80px 위까지
        self.ROI_HEIGHT = 240          # 아래에서 240px 지점부터 ROI

        # 신규 객체 판별을 위한 이전 프레임 박스들 저장
        self.previous_boxes = []       # [(x1,y1,x2,y2,class), ...]

        # 판단 기준
        self.IOU_TH = 0.3              # IoU threshold
        self.DIST_TH = 40              # 중심점 거리 threshold(px 단위)

    # -------------------------
    # IoU 계산
    # -------------------------
    def compute_iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union

    # -------------------------
    # 중심점 거리 계산
    # -------------------------
    def compute_distance(self, b1, b2):
        cx1 = (b1[0] + b1[2]) / 2
        cy1 = (b1[1] + b1[3]) / 2
        cx2 = (b2[0] + b2[2]) / 2
        cy2 = (b2[1] + b2[3]) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    # -------------------------
    # 메인 YOLO 감지 함수
    # -------------------------
    def detect_objects(self, frame, frame_idx, detection_interval=1):
        if frame_idx % detection_interval != 0:
            return None, None

        h, w, _ = frame.shape

        # ROI 계산
        roi_y1 = max(0, h - self.ROI_HEIGHT)
        roi_y2 = max(0, h - self.ROI_BOTTOM_MARGIN)
        roi = frame[roi_y1:roi_y2, :]

        # YOLO inference
        results = self.model(roi, conf=self.conf, verbose=False)[0]

        detections = []
        if results.boxes:
            xyxy = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, cls in zip(xyxy, clss):
                x1, y1, x2, y2 = box

                # ROI → 원본 좌표
                y1 += roi_y1
                y2 += roi_y1

                detections.append([x1, y1, x2, y2, int(cls)])

        annotated = frame.copy()
        detected_counts = defaultdict(int)
        current_frame_boxes = []

        # ArUco 초기 처리
        if self.aruco_detector and self.aruco_detector.get_current_point() is None:
            self.aruco_detector.set_current_point("Alpha")

        # -------------------------
        # 신규 객체 판별
        # -------------------------
        for det in detections:
            x1, y1, x2, y2, cls = det
            class_name = self.object_names[cls]

            is_new = True

            for prev in self.previous_boxes:
                px1, py1, px2, py2, pcls = prev

                # IoU 계산
                iou = self.compute_iou(det[:4], prev[:4])

                # 중심 거리 계산
                dist = self.compute_distance(det[:4], prev[:4])

                # 동일 객체로 판단되는 조건
                if iou > self.IOU_TH and dist < self.DIST_TH:
                    is_new = False
                    break

            # 신규 객체이면 count + 1
            if is_new:
                detected_counts[class_name] += 1

                if self.aruco_detector:
                    self.aruco_detector.update_detection(class_name)

            # 이번 프레임 박스 저장
            current_frame_boxes.append([x1, y1, x2, y2, cls])

            # 박스 시각화
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"{class_name}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ROI 시각화
        cv2.rectangle(annotated, (0, roi_y1), (w, roi_y2), (255, 0, 0), 2)

        # 이전 박스 갱신
        self.previous_boxes = current_frame_boxes

        return annotated, detected_counts
