"""
LED 제어 모듈
"""
from tiki.mini import TikiMini


class LEDController:
    """LED 제어 클래스"""
    
    def __init__(self, tiki: TikiMini):
        """LED 제어기 초기화
        
        Args:
            tiki: TikiMini 로봇 객체
        """
        self.tiki = tiki
        
        # QR 코드와 LED 패턴 매핑
        self.qr_mapping = {
            'ID_O_R': ('O', (50, 0, 0)),
            'ID_O_G': ('O', (0, 50, 0)),
            'ID_O_B': ('O', (0, 0, 50)),
            'ID_X_R': ('X', (50, 0, 0)),
            'ID_X_G': ('X', (0, 50, 0)),
            'ID_X_B': ('X', (0, 0, 50)),
            'ID_#_R': ('#', (50, 0, 0)),
            'ID_#_G': ('#', (0, 50, 0)),
            'ID_#_B': ('#', (0, 0, 50)),
        }
    
    def clear_leds(self):
        """모든 LED 끄기"""
        for i in range(16):
            self.tiki.set_led(0, i, 0, 0, 0)
    
    def draw_shape(self, shape, color):
        """LED에 도형 그리기
        
        Args:
            shape: 도형 종류 ('X', 'O', '#')
            color: RGB 색상 튜플 (r, g, b)
        """
        self.clear_leds()
        r, g, b = color
        
        if shape == 'X':
            indices = [15, 0, 9, 6, 10, 5, 12, 3]
        elif shape == 'O':
            indices = [15, 8, 7, 0, 12, 11, 4, 3, 14, 13, 1, 2]
        elif shape == '#':
            indices = list(set([8, 9, 10, 11, 7, 6, 5, 4, 14, 9, 6, 1, 13, 10, 5, 2]))
        else:
            return
        
        for i in indices:
            self.tiki.set_led(0, i, r, g, b)
    
    def handle_qr_code(self, qr_data):
        """QR 코드 데이터를 받아서 LED 패턴 표시
        
        Args:
            qr_data: QR 코드에서 읽은 데이터 문자열
            
        Returns:
            bool: 매핑된 QR 코드인지 여부
        """
        if qr_data in self.qr_mapping:
            shape, color = self.qr_mapping[qr_data]
            self.draw_shape(shape, color)
            print(f"[LED] QR 코드 인식: {qr_data} → {shape} 패턴 표시")
            return True
        return False

