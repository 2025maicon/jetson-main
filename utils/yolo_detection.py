import cv2
import numpy as np
from collections import defaultdict
from utils.config import OBJECT_NAMES


class SimpleTracker:
    def __init__(self, iou_th=0.2):
        self.iou_th = iou_th
        self.last_boxes = []
        self.track_ids = []
        self.next_id = 0

    def update(self, detections):
        new_track_ids = []
        results = []

        for det in detections:
            box = det[:4]
            cls = int(det[5])

            matched = False
            match_id = None

            for idx, prev in enumerate(self.last_boxes):
                if self.iou(box, prev) > self.iou_th:
                    matched = True
                    match_id = self.track_ids[idx]
                    break

            if not matched:
                match_id = self.next_id
                self.next_id += 1

            new_track_ids.append(match_id)
            results.append((match_id, cls, box))

        self.last_boxes = [d[:4] for d in detections]
        self.track_ids = new_track_ids

        return results

    def iou(self, b1, b2):
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


class YOLODetector:
    """YOLO + SimpleTracker 기반 객체 감지 + ROI 적용"""

    def __init__(self, model, aruco_detector=None, conf=0.45):
        self.model = model
        self.aruco_detector = aruco_detector
        self.object_names = OBJECT_NAMES
        self.conf = conf
        self.tracker = SimpleTracker(iou_th=1.5)
        self.counted_ids = set()

        # ROI 하단 기준 (아래에서 20~80 px)
        self.ROI_BOTTOM_MARGIN = 80
        self.ROI_HEIGHT = 240

    def detect_objects(self, frame, frame_idx, detection_interval=1):
        if frame_idx % detection_interval != 0:
            return None, None

        h, w, _ = frame.shape

        # ROI 범위 계산
        roi_y1 = max(0, h - self.ROI_HEIGHT)      # 아래에서 80px
        roi_y2 = max(0, h - self.ROI_BOTTOM_MARGIN)  # 아래에서 20px

        # ROI 추출
        roi = frame[roi_y1:roi_y2, :]

        # YOLO 감지
        results = self.model(roi, conf=self.conf, verbose=False)[0]

        detections = []
        if results.boxes:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, score, cls in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box

                # ROI → 원본 좌표 복구
                y1 += roi_y1
                y2 += roi_y1

                detections.append([x1, y1, x2, y2, float(score), int(cls)])

        tracks = self.tracker.update(detections)

        detected_counts = defaultdict(int)
        annotated = frame.copy()

        # ArUco point 초기화
        if self.aruco_detector and self.aruco_detector.get_current_point() is None:
            self.aruco_detector.set_current_point("Alpha")

        # 시각화 및 카운트
        for track_id, cls, box in tracks:
            class_name = self.object_names[cls]
            x1, y1, x2, y2 = box

            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                detected_counts[class_name] += 1
                if self.aruco_detector:
                    self.aruco_detector.update_detection(class_name)

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(annotated, f"{class_name} ID:{track_id}",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ROI 영역 표시(디버그용)
        cv2.rectangle(
            annotated,
            (0, roi_y1),
            (w, roi_y2),
            (255, 0, 0),
            2
        )

        return annotated, detected_counts
