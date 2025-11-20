"""
주행 제어 모듈
"""
from utils.config import BASE_SPEED, MAX_STEERING, FRAME_CENTER_X


def control_motors(tiki, center_x, is_horizontal_bar, pid_controller):
    """차선 중앙을 기준으로 모터 제어
    
    Args:
        tiki: TikiMini 로봇 객체
        center_x: 감지된 차선 중앙 X 좌표 (None일 수 있음)
        is_horizontal_bar: 가로 흰줄 감지 여부
        pid_controller: PIDController 객체
    """
    if center_x is not None:
        error = center_x - FRAME_CENTER_X
        
        steering = pid_controller.compute(error)
        steering *= 0.5
        steering = float(max(-MAX_STEERING, min(MAX_STEERING, steering)))
        
        if is_horizontal_bar:
            steering = 0
        
        L = int(max(0, min(127, BASE_SPEED + steering)))
        Rm = int(max(0, min(127, BASE_SPEED - steering)))
        
        tiki.set_motor_power(tiki.MOTOR_LEFT, L)
        tiki.set_motor_power(tiki.MOTOR_RIGHT, Rm)
    else:
        tiki.stop()

