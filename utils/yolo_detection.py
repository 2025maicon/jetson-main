"""
YOLO 객체 감지 및 처리 모듈
"""
import cv2
from collections import defaultdict
from utils.config import OBJECT_NAMES


class YOLODetector:
    """YOLO 객체 감지 및 처리 클래스"""
    
    def __init__(self, model, aruco_detector=None):
        """YOLO 감지기 초기화
        
        Args:
            model: YOLO 모델 객체
            aruco_detector: ArUcoDetector 객체 (current_point 업데이트용)
        """
        self.model = model
        self.aruco_detector = aruco_detector
        self.object_names = OBJECT_NAMES
    
    def detect_objects(self, frame, frame_idx, detection_interval=5):
        """프레임에서 객체 감지
        
        Args:
            frame: 입력 프레임
            frame_idx: 현재 프레임 인덱스
            detection_interval: 감지 간격 (N프레임마다 수행)
            
        Returns:
            tuple: (annotated_frame, detected_counts) 또는 (None, None)
        """
        if frame_idx % detection_interval != 0:
            return None, None
        
        frame_for_yolo = frame.copy()
        results = self.model(frame_for_yolo, conf=0.45)
        
        detected_counts = defaultdict(int)
        
        if results and len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.cpu().numpy()
            
            # 객체가 인식됐을 때 current_point가 None이면 첫번째 point인 Alpha로 설정
            if self.aruco_detector and self.aruco_detector.get_current_point() is None:
                self.aruco_detector.set_current_point("Alpha")
                print(f"[DETECTION] current_point가 None이었으므로 Alpha로 설정")
            
            for cls in classes:
                class_name = self.object_names[int(cls)]
                detected_counts[class_name] += 1
                
                # 현재 point에 detection 정보 누적
                if self.aruco_detector:
                    self.aruco_detector.update_detection(class_name)
            
            if len(detected_counts) > 0:
                current_point = self.aruco_detector.get_current_point() if self.aruco_detector else None
                print(f"[DETECTION] 객체 감지: {dict(detected_counts)} (현재 point: {current_point})")
        
        # 시각화
        annotated = results[0].plot()
        cv2.putText(
            annotated,
            "Object Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        
        return annotated, detected_counts

