"""
설정 상수 및 전역 변수
"""
import numpy as np

# ArUco 마커 매핑
MARKER_TO_POINT = {
    1:  { 'kind': 'point',  'name': 'Alpha' },
    2:  { 'kind': 'sector', 'name': 'sector1' },
    3:  { 'kind': 'sector', 'name': 'sector2' },
    4:  { 'kind': 'sector', 'name': 'sector3' },
    5:  { 'kind': 'sector', 'name': 'sector4' },
    6:  { 'kind': 'sector', 'name': 'sector5' },
    7:  { 'kind': 'sector', 'name': 'sector6' },
    8:  { 'kind': 'point',  'name': 'Bravo' },
    9:  { 'kind': 'sector', 'name': 'sector7' },
    10: { 'kind': 'point',  'name': 'Charlie' },
    11: { 'kind': 'sector', 'name': 'sector8' },
    12: { 'kind': 'sector', 'name': 'sector9' },
    13: { 'kind': 'point',  'name': 'Finish' },
}

# 객체 이름 리스트
OBJECT_NAMES = ['box', 'car', 'enemy', 'hazmat', 'missile', 'mortar', 'tank']

# 주행 파라미터
BASE_SPEED = 70
MAX_STEERING = 27
FRAME_CENTER_X = 160  # 320x240 기준 중앙 X

# Bird's-Eye View 설정
BIRD_WIDTH, BIRD_HEIGHT = 320, 240

SRC_POINTS = np.float32([
    [60, 120],
    [260, 120],
    [310, 230],
    [10, 230],
])

DST_POINTS = np.float32([
    [0, 0],
    [BIRD_WIDTH - 1, 0],
    [BIRD_WIDTH - 1, BIRD_HEIGHT - 1],
    [0, BIRD_HEIGHT - 1],
])

# 포트홀 감지 파라미터
POTHOLE_DETECT_FRAMES = 4       # 연속 판정 프레임 수 (감도)
POTHOLE_COOLDOWN = 60.0         # 회피 후 재감지 대기(sec)
POTHOLE_MIN_ABS = 0.01          # 절대 최소 흰색 비율 (너무 작은 값이면 무시)
POTHOLE_RATIO = 0.4             # 기준(EMA) 대비 이 비율보다 작으면 포트홀로 판단
EMA_ALPHA = 0.05                # EMA 업데이트 계수 (기준 적응 속도)

# JSON 파일 경로
JSON_PATH = "/home/jetson/Workspace/report/A3R8.json"

# 서버 URL
SERVER_URL = "http://58.229.150.23:5000"

