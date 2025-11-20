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

            elif info['kind'] == 'sector':
                sector_detected = True
                sector_name = info['name']
                self._handle_sector(info, frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        return point_detected, sector_detected, sector_name


    # --------------------------------------------------------
    # POINT 처리 함수
    # --------------------------------------------------------
    def _handle_point(self, info, frame):
        point_name = info['name']
        self.current_point = point_name

        if point_name not in self.visited_points:
            self.visited_points.add(point_name)
            self.transmitted_points.append(point_name)

            detection_for_payload = {}
            for pt in self.transmitted_points:
                detection_list = []
                if pt in self.point_detections:
                    for class_name, count in self.point_detections[pt].items():
                        if count > 0:
                            detection_list.append({"type": class_name, "count": count})
                detection_for_payload[pt] = detection_list

            print(f"[POINT] {point_name} 최초 통과 → 서버 전송")
            send_to_server(points=self.transmitted_points, detected_objects=detection_for_payload)

            return True

        return False


    # --------------------------------------------------------
    # SECTOR 처리 함수
    # --------------------------------------------------------
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

                num = re.search(r"\d+", sector_name)
                if num:
                    sector_num = num.group()

                    filename = f"{mission_code}_sector{sector_num}.jpg"
                    image_path = os.path.join("/tmp", filename)

                    cv2.imwrite(image_path, frame)
                    print(f"[FIRE] {sector_name} → 이미지 캡처 {filename}")

                    send_dashboard_image(image_path)

        except Exception as e:
            print(f"[ERROR] sector 처리 중 오류: {e}")


    # --------------------------------------------------------
    # ArUco detect() (전처리 포함)
    # --------------------------------------------------------
    def detect(self, frame):
        h, w, _ = frame.shape

        roi_y1 = h // 2
        roi = frame[roi_y1:h, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        thresh = cv2.adaptiveThreshold(
            sharpen, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            2
        )

        corners, ids, rejected = self.detector.detectMarkers(thresh)

        # ROI offset 복구
        if corners is not None:
            for c in corners:
                c[:, :, 1] += roi_y1

        return corners, ids, rejected


    def update_detection(self, class_name):
        if self.current_point is not None:
            self.point_detections[self.current_point][class_name] += 1

    def set_current_point(self, point_name):
        self.current_point = point_name

    def get_current_point(self):
        return self.current_point
