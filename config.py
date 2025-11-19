# config.py
"""
프로젝트 전체에서 공통으로 사용하는 설정값들.
- YOLO 가중치 경로
- 카메라 파이프라인
- 차선 인식 파라미터 등
"""

# YOLOv8n 학습된 가중치 경로 (Jetson에 복사해 둘 것)
YOLO_WEIGHTS_PATH = "models/road_v2_best.pt"

# 데이터셋 클래스 이름들 (Roboflow data.yaml 기준으로 맞춰서 수정)
CLASS_NAMES = [
    "hazmat",
    "missile",
    "enemy",
    "tank",
    "car",
    "mortar",
    "box",
    # TODO: 실제 클래스 이름에 맞게 채우기
]

# Jetson 카메라 GStreamer 파이프라인 (노트북에서 쓰던 것 재사용)
CAMERA_PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=320, height=240, format=NV12, framerate=60/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

# 차선 인식용 HSV 범위 (흰색)
HSV_WHITE_LOWER = (0, 0, 200)
HSV_WHITE_UPPER = (180, 30, 255)
ROI_RATIO = 0.5  # 아래쪽 50%만 사용

# 주행 제어 관련 (대략 값, 나중에 튜닝)
BASE_SPEED = 0.3     # 기본 속도 (0~1 기준이라고 가정)
TURN_GAIN = 0.4      # 조향 보정 강도
