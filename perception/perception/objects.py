# perception/objects.py
"""
YOLOv8n을 이용해 객체를 탐지하고,
(선택) 추적 ID(track_id)를 함께 반환하는 모듈.
"""

from ultralytics import YOLO
from config import YOLO_WEIGHTS_PATH, CLASS_NAMES


# 모듈 import 시 한 번만 모델 로드
model = YOLO(YOLO_WEIGHTS_PATH)

# 클래스 ID -> 이름 매핑 (Ultralytics 모델 내부 names 우선, 없으면 config 사용)
MODEL_NAMES = model.model.names if hasattr(model, "model") else {
    i: name for i, name in enumerate(CLASS_NAMES)
}


def detect_and_track_objects(frame, conf=0.5, iou=0.5):
    """
    frame: BGR 이미지 (numpy.ndarray)

    return:
        detections: dict 리스트
        [
          {
            "cls_id": int,
            "cls_name": str,
            "conf": float,
            "bbox": (x1, y1, x2, y2),
            "track_id": int or None,
          },
          ...
        ]
    """
    results = model.track(
        source=frame,
        imgsz=640,
        conf=conf,
        iou=iou,
        persist=True,   # 같은 객체에 같은 track_id 유지
        verbose=False
    )[0]

    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        track_id = None
        if box.id is not None:
            track_id = int(box.id[0])

        cls_name = MODEL_NAMES.get(cls_id, f"cls_{cls_id}")

        detections.append(
            {
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "track_id": track_id,
            }
        )

    return detections
