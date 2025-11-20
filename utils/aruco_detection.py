"""
ArUco 마커 감지 및 처리 모듈
"""
import cv2
import json
import os
import re
from collections import defaultdict
from utils.config import MARKER_TO_POINT, JSON_PATH
from utils.server_communication import send_to_server, send_dashboard_image


class ArUcoDetector:
    """ArUco 마커 감지 및 처리 클래스"""
    
    def __init__(self):
        """ArUco 감지기 초기화"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.marker_to_point = MARKER_TO_POINT
        
        # 상태 변수
        self.visited_points = set()
        self.transmitted_points = []
        self.visited_sectors = set()
        self.current_point = None
        self.point_detections = defaultdict(lambda: defaultdict(int))
    
    def process_markers(self, frame, ids, corners):
        """ArUco 마커 처리 및 point/sector 처리"""

        if ids is None:
            return False, False, None

        ids_list = [int(i[0]) for i in ids]

        point_detected = False
        sector_detected = False
        sector_name = None

        for marker_id in ids_list:
            if marker_id not in self.marker_to_point:
                continue
            
            info = self.marker_to_point[marker_id]

            if info['kind'] == 'point':
                point_detected = self._handle_point(info, frame)

            else:
                sector_detected = True
                sector_name = info['name']
                self._handle_sector(info, frame)   # JSON 로직만 처리

        # 시각화
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        return point_detected, sector_detected, sector_name

    
    def _handle_sector(self, info, frame):
        sector_name = info['name']
        print(f"[SECTOR] 감지: {sector_name}")

        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            fire_buildings = json_data.get("fire_buildings", [])
            mission_code = json_data.get("mission_code", "A3R8")

            if sector_name in fire_buildings and sector_name not in self.visited_sectors:
                self.visited_sectors.add(sector_name)

                # 이미지 촬영/전송은 하지 않음
                # → main에서 처리
                print(f"[FIRE BUILDING] {sector_name} 감지 → 회전 및 촬영은 main에서 수행")

        except Exception as e:
            print(f"[ERROR] fire_building 확인 중 오류: {e}")

    def detect(self, frame):
        """
        최적화된 ArUco 마커 감지 (ROI + CLAHE + Sharpen + AdaptiveThreshold)
        Returns:
            corners, ids, rejected
        """

        h, w, _ = frame.shape

        # -------------------------------------------
        # 1) ROI 설정 (하단 50%만 검사) → 인식률/속도 향상
        # -------------------------------------------
        roi_y1 = h // 2
        roi = frame[roi_y1:h, :]

        # -------------------------------------------
        # 2) 그레이스케일
        # -------------------------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------
        # 3) CLAHE (대비 증가 → 작은 마커 검출력 크게 향상)
        # -------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # -------------------------------------------
        # 4) GaussianBlur (노이즈 제거 → threshold 안정성 증가)
        # -------------------------------------------
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # -------------------------------------------
        # 5) Sharpen (작은 마커 edge 강화)
        # -------------------------------------------
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        # -------------------------------------------
        # 6) Adaptive Threshold (조명·반사에 가장 강력)
        # -------------------------------------------
        thresh = cv2.adaptiveThreshold(
            sharpen,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,      # blockSize
            2        # C값
        )

        # -------------------------------------------
        # 7) ArUco detect (전처리된 이미지)
        # -------------------------------------------
        corners, ids, rejected = self.detector.detectMarkers(thresh)

        # -------------------------------------------
        # 8) ROI → 원본 좌표 복구 (draw는 원본 frame에서)
        # -------------------------------------------
        if corners is not None:
            for c in corners:
                c[:, :, 1] += roi_y1  # y좌표만 ROI offset 보정

        return corners, ids, rejected


    def update_detection(self, class_name):
        """현재 point에 detection 정보 누적"""
        if self.current_point is not None:
            self.point_detections[self.current_point][class_name] += 1
    
    def set_current_point(self, point_name):
        """현재 point 설정 (YOLO 감지 시 사용)"""
        self.current_point = point_name
    
    def get_current_point(self):
        """현재 point 반환"""
        return self.current_point

