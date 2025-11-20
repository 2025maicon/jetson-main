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
        """ArUco 마커 처리 및 point/sector 처리
        
        Args:
            frame: 원본 프레임
            ids: 감지된 마커 ID 리스트
            corners: 마커 코너 좌표
            
        Returns:
            bool: point가 감지되어 서버로 전송되었는지 여부
        """
        if ids is None:
            return False
        
        ids_list = [int(i[0]) for i in ids]
        point_detected = False
        
        for marker_id in ids_list:
            if marker_id not in self.marker_to_point:
                continue
                
            info = self.marker_to_point[marker_id]
            
            if info['kind'] == 'point':
                point_detected = self._handle_point(info, frame)
            else:
                self._handle_sector(info, frame)
        
        # 마커 시각화
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        return point_detected
    
    def _handle_point(self, info, frame):
        """Point 마커 처리"""
        point_name = info['name']
        self.current_point = point_name
        
        if point_name not in self.visited_points:
            self.visited_points.add(point_name)
            self.transmitted_points.append(point_name)
            
            # 모든 point에 대해 누적 detection 정보 변환
            detection_for_payload = {}
            for pt in self.transmitted_points:
                detection_list = []
                if pt in self.point_detections:
                    for class_name, count in self.point_detections[pt].items():
                        if count > 0:
                            detection_list.append({"type": class_name, "count": count})
                detection_for_payload[pt] = detection_list
            
            print(f"[POINT] {point_name} 최초 통과 → 대시보드로 전송 (points 업데이트, detection: {detection_for_payload})")
            send_to_server(points=self.transmitted_points, detected_objects=detection_for_payload)
            return True
        
        return False
    
    def _handle_sector(self, info, frame):
        """Sector 마커 처리 (fire_building 이미지 캡처)"""
        sector_name = info['name']
        print(f"[SECTOR] 감지: {sector_name}")
        
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            fire_buildings = json_data.get("fire_buildings", [])
            mission_code = json_data.get("mission_code", "A3R8")
            
            # fire_buildings에 포함되어 있고, 아직 전송하지 않은 sector인 경우
            if sector_name in fire_buildings and sector_name not in self.visited_sectors:
                self.visited_sectors.add(sector_name)
                
                # sector 이름에서 숫자 추출 (예: "sector8" -> "8")
                sector_number = re.search(r'\d+', sector_name)
                if sector_number:
                    section_num = sector_number.group()
                    
                    # 이미지 파일 이름 생성: A3R8_section8.jpg
                    image_filename = f"{mission_code}_sector{section_num}.jpg"
                    image_path = os.path.join("/tmp", image_filename)
                    
                    # 현재 프레임 캡처
                    cv2.imwrite(image_path, frame)
                    print(f"[FIRE BUILDING] {sector_name} 감지 → 이미지 캡처: {image_filename}")
                    
                    # 서버로 이미지 전송
                    send_dashboard_image(image_path)
                    
                    # 임시 파일 삭제
                    # try:
                    #     os.remove(image_path)
                    # except:
                    #     pass
                else:
                    print(f"[WARNING] sector 이름에서 숫자를 추출할 수 없습니다: {sector_name}")
        except FileNotFoundError:
            print(f"[WARNING] JSON 파일을 찾을 수 없습니다: {JSON_PATH}")
        except Exception as e:
            print(f"[ERROR] fire_building 확인 중 오류: {e}")
    
    def detect(self, frame):
        """
        전처리 기반 ArUco 감지 (Nano 최적화)
        Returns:
            corners, ids, rejected
        """
        # 1) Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2) 대비 증가 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 3) 노이즈 감소
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4) Adaptive Threshold
        # 조명 변화가 있거나 반사광이 있을 때 검출 안정성이 가장 좋아짐
        thresh = cv2.adaptiveThreshold(gray,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    15,
                                    2)

        # 5) ArUco detect 수행 (전처리된 이미지)
        corners, ids, rejected = self.detector.detectMarkers(thresh)

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

