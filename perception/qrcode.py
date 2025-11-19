# perception/qrcode.py
"""
QR 코드를 인식해서 문자열을 반환하는 모듈.
"""

import cv2

_qr_detector = cv2.QRCodeDetector()


def detect_qr_data(frame):
    """
    frame: BGR 이미지

    return:
        data (str) 또는 None
    """
    data, points, _ = _qr_detector.detectAndDecode(frame)
    if data:
        return data
    return None
