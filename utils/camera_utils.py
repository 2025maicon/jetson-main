"""
카메라 초기화 및 설정 모듈
"""
import cv2
from utils.config import BIRD_WIDTH, BIRD_HEIGHT, SRC_POINTS, DST_POINTS


def init_camera():
    """GStreamer 파이프라인으로 카메라 초기화
    
    Returns:
        cv2.VideoCapture: 초기화된 카메라 객체
    """
    pipeline = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=60/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")
    
    # 최신 프레임만 사용하도록 버퍼 최소화
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap


def init_perspective_transform():
    """Bird's-Eye View를 위한 투시 변환 행렬 생성
    
    Returns:
        numpy.ndarray: 투시 변환 행렬
    """
    return cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

