# perception/aruco.py
"""
Aruco 마커를 인식해서 ID를 반환하는 모듈.
"""

import cv2


# OpenCV에서 제공하는 기본 딕셔너리 사용 (필요에 따라 변경)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


def detect_aruco_id(frame):
    """
    frame: BGR 이미지

    return:
        marker_id (int) 또는 None
        (여러 개가 보이면 일단 첫 번째 것만 반환)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # ids는 shape (N, 1)의 배열
    marker_id = int(ids[0][0])
    return marker_id
