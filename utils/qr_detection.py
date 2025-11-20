"""
QR 코드 감지 모듈
"""
import cv2
import time


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
    
    def detect_from_camera(self, cap, led_controller=None, max_attempts=10, 
                          stabilization_time=0.5, attempt_interval=0.2):
        """카메라에서 QR 코드 인식 (주행 시작 지점용)
        
        Args:
            cap: OpenCV VideoCapture 객체
            led_controller: LEDController 객체 (선택사항, 제공 시 자동으로 LED 제어)
            max_attempts: 최대 시도 횟수 (기본값: 10)
            stabilization_time: 카메라 안정화 대기 시간(초) (기본값: 0.5)
            attempt_interval: 각 시도 간 간격(초) (기본값: 0.2)
            
        Returns:
            str: 인식된 QR 코드 데이터 (실패 시 빈 문자열)
        """
        print("[QR] 주행 시작 지점에서 QR 코드 인식 시작...")
        
        # 카메라가 안정화될 때까지 대기
        time.sleep(stabilization_time)
        
        # QR 코드 인식 시도
        for attempt in range(max_attempts):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # 카메라 기준 회전 보정
            frame = cv2.flip(frame, -1)
            
            # QR 코드 인식
            qr_data, bbox, _ = self.detect(frame)
            
            if qr_data:
                print(f"[QR] QR 코드 인식 성공: {qr_data}")
                
                # LED 제어기가 제공된 경우 자동으로 LED 제어
                if led_controller is not None:
                    if led_controller.handle_qr_code(qr_data):
                        print(f"[LED] QR 코드에 맞는 LED 패턴 표시 완료")
                    else:
                        print(f"[LED] 알 수 없는 QR 코드: {qr_data}")
                
                return qr_data
            else:
                print(f"[QR] 시도 {attempt + 1}/{max_attempts}: QR 코드 미인식, 재시도...")
                time.sleep(attempt_interval)
        
        print("[QR] QR 코드 인식 실패 (주행 계속 진행)")
        return ""
    
    def detect_from_frame(self, frame, led_controller=None):
        """프레임에서 QR 코드 인식 및 LED 제어 (메인 루프용)
        
        Args:
            frame: 입력 프레임 (BGR, 이미 회전 보정된 프레임)
            led_controller: LEDController 객체 (선택사항, 제공 시 자동으로 LED 제어)
            
        Returns:
            str: 인식된 QR 코드 데이터 (없으면 빈 문자열)
        """
        # QR 코드 인식
        qr_data, bbox, _ = self.detect(frame)
        
        if qr_data:
            print(f"[QR] QR 코드 인식 성공: {qr_data}")
            
            # LED 제어기가 제공된 경우 자동으로 LED 제어
            if led_controller is not None:
                if led_controller.handle_qr_code(qr_data):
                    print(f"[LED] QR 코드에 맞는 LED 패턴 표시 완료")
                else:
                    print(f"[LED] 알 수 없는 QR 코드: {qr_data}")
            
            return qr_data
        
        return ""

