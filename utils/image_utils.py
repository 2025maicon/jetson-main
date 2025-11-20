"""
이미지 유틸리티 모듈
"""
import cv2


def frame_to_bytes_drive(frame):
    """주행 디버그 패널 전송용 (낮은 품질, 부하 적음)"""
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    return buf.tobytes()


def frame_to_bytes_yolo(frame):
    """YOLO/ArUco 시각화용 (더 높은 품질, 사람이 보기 좋게)"""
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # 필요시 60~80 사이에서 조절
    return buf.tobytes()

