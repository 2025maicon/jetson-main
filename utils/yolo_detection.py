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
        """
        detections: [[x1,y1,x2,y2,score,class], ...]
        Returns: list of (track_id, class, box)
        """
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
    """YOLO + SimpleTracker 기반 객체 감지 및 처리"""

    def __init__(self, model, aruco_detector=None, conf=0.45):
        self.model = model
        self.aruco_detector = aruco_detector
        self.object_names = OBJECT_NAMES
        self.conf = conf

        # 초경량 트래커
        self.tracker = SimpleTracker(iou_th=0.2)

        # track_id 중복 카운트 방지
        self.counted_ids = set()

    def detect_objects(self, frame, frame_idx, detection_interval=1):
        if frame_idx % detection_interval != 0:
            return None, None

        frame_for_yolo = frame.copy()
        results = self.model(frame_for_yolo, conf=self.conf, verbose=False)[0]

        # YOLO → tracker 입력
        detections = []
        if results.boxes:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, score, cls in zip(xyxy, confs, clss):
                detections.append([*box, float(score), int(cls)])

        tracks = self.tracker.update(detections)

        detected_counts = defaultdict(int)
        annotated = frame.copy()

        # AruCo 초기화
        if self.aruco_detector and self.aruco_detector.get_current_point() is None:
            self.aruco_detector.set_current_point("Alpha")

        for track_id, cls, box in tracks:
            class_name = self.object_names[cls]
            x1, y1, x2, y2 = box

            # 신규 등장한 track_id만 카운트
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                detected_counts[class_name] += 1

                if self.aruco_detector:
                    self.aruco_detector.update_detection(class_name)

            # 시각화
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                annotated,
                f"{class_name} ID:{track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return annotated, detected_counts