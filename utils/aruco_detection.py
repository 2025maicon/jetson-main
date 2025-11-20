"""
ArUco 마커 감지 및 처리 모듈 (최적화 버전)
"""
import cv2
import json
import os
import re
from collections import defaultdict
from utils.config import MARKER_TO_POINT, JSON_PATH
from utils.server_communication import send_to_server, send_dashboard_image


class ArUcoDetector:
    """최적화된 ArUco 마커 감지 및 처리 클래스"""

    def __init__(self):
        """ArUco 감지기 초기화"""

        # 기본 딕셔너리
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # 🔥 최적화된 DetectorParameters
        params = cv2.aruco.DetectorParameters()

        # ---- corner refinement ----
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30

        # ---- adaptive threshold ----
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.minCornerDistanceRate = 0.05
        params.minOtsuStdDev = 5.0

        # 검출기 생성
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self.marker_to_point = MARKER_TO_POINT
        
        # 상태 변수
        self.visited_points = set()
        self.transmitted_points = []
        self.visited_sectors = set()
        self.current_point = None
        self.point_detections = defaultdict(lambda: defaultdict(int))

    # ============================================================
    # 🔥 최적화된 detect() : CLAHE + Blur 적용
    # ============================================================
    def detect(self, frame):
        """전처리 후 ArUco 마커 감지"""

        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 대비 강화(CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 약간의 blur로 노이즈 감소
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected

    # ============================================================
    # 기존 process_markers / handle_point / handle_sector 유지
    # ============================================================
    def process_markers(self, frame, ids, corners):
        """ArUco 마커 처리 및 point/sector 처리"""

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

        # 시각화
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

            # 전송용 detection 정리
            detection_for_payload = {}
            for pt in self.transmitted_points:
                detection_list = []
                if pt in self.point_detections:
                    for class_name, count in self.point_detections[pt].items():
                        if count > 0:
                            detection_list.append({"type": class_name, "count": count})
                detection_for_payload[pt] = detection_list

            print(f"[POINT] {point_name} 최초 통과 → 대시보드 전송")
            send_to_server(points=self.transmitted_points, detected_objects=detection_for_payload)
            return True

        return False

    def _handle_sector(self, info, frame):
        """Sector 마커 처리: fire_building 이미지 캡처"""

        sector_name = info['name']
        print(f"[SECTOR] 감지: {sector_name}")

        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            fire_buildings = json_data.get("fire_buildings", [])
            mission_code = json_data.get("mission_code", "A3R8")

            if sector_name in fire_buildings and sector_name not in self.visited_sectors:
                self.visited_sectors.add(sector_name)

                # sector 번호 추출
                sector_number = re.search(r'\d+', sector_name)
                if sector_number:
                    section_num = sector_number.group()

                    image_filename = f"{mission_code}_section{section_num}.jpg"
                    image_path = os.path.join("/tmp", image_filename)

                    # 이미지 저장
                    cv2.imwrite(image_path, frame)
                    print(f"[FIRE BUILDING] {sector_name} → 이미지 캡처: {image_filename}")

                    send_dashboard_image(image_path)

                    # 임시 파일 삭제
                    try:
                        os.remove(image_path)
                    except:
                        pass
                else:
                    print(f"[WARNING] sector 이름에서 숫자 추출 실패: {sector_name}")

        except FileNotFoundError:
            print(f"[WARNING] JSON 파일 없음: {JSON_PATH}")
        except Exception as e:
            print(f"[ERROR] fire_building 처리 오류: {e}")

    def update_detection(self, class_name):
        """현재 point에 detection 정보 누적"""
        if self.current_point is not None:
            self.point_detections[self.current_point][class_name] += 1

    def set_current_point(self, point_name):
        """현재 point 설정"""
        self.current_point = point_name

    def get_current_point(self):
        """현재 point 반환"""
        return self.current_point
