"""
QR 코드 감지 모듈
"""
import cv2


class QRDetector:
    """QR 코드 감지 클래스"""
    
    def __init__(self):
        """QR 코드 감지기 초기화"""
        self.detector = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.last_detected_qr = None  # 마지막으로 감지된 QR 코드 (중복 방지)
    
    def detect(self, frame):
        """프레임에서 QR 코드 감지
        
        Args:
            frame: 입력 프레임 (BGR)
            
        Returns:
            tuple: (qr_data, bbox, visualized_frame)
                - qr_data: QR 코드 데이터 (없으면 빈 문자열)
                - bbox: QR 코드 바운딩 박스 좌표 (없으면 None)
                - visualized_frame: QR 코드가 시각화된 프레임
        """
        visualized_frame = frame.copy()
        qr_data = ""
        bbox = None
        
        # 방법 1: 원본 이미지로 시도
        data, bbox, _ = self.detector.detectAndDecode(frame)
        
        # 방법 2: 실패 시 그레이스케일 + CLAHE로 재시도
        if not data:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(gray)
            data, bbox, _ = self.detector.detectAndDecode(enhanced)
        
        if data:
            qr_data = data
            
            # QR 코드 시각화
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(visualized_frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
                cv2.putText(visualized_frame, qr_data, (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return qr_data, bbox, visualized_frame
    
    def detect_new(self, frame):
        """새로운 QR 코드만 감지 (중복 방지)
        
        Args:
            frame: 입력 프레임
            
        Returns:
            str: 새로운 QR 코드 데이터 (없거나 중복이면 빈 문자열)
        """
        qr_data, bbox, _ = self.detect(frame)
        
        # 새로운 QR 코드인 경우에만 반환
        if qr_data and qr_data != self.last_detected_qr:
            self.last_detected_qr = qr_data
            return qr_data
        
        return ""

