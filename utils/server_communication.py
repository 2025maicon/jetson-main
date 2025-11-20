"""
서버 통신 모듈
"""
import json
import os
import requests
from pathlib import Path
from utils.config import JSON_PATH, SERVER_URL


def send_to_server(point=None, detected_objects=None, points=None, fire_buildings=None):
    """Send dashboard payload. Use `points` to pass Alpha/Bravo/Charlie list.
    - `point` kept for backward compatibility (not used for point-list payloads)
    - `detected_objects` expects a dict (detection map)
    - `points` is a list of point names (capitalized) to include in payload
    - `fire_buildings` is an optional list
    - Loads base JSON from /home/jetson/Workspace/report/A3R8.json
    - Only updates `points` and `detection`, keeps `mission_code` and `fire_buildings` from file
    """
    # JSON 파일에서 기본 데이터 로드
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            base_payload = json.load(f)
    except FileNotFoundError:
        print(f"[WARNING] JSON 파일을 찾을 수 없습니다: {JSON_PATH}, 기본값 사용")
        base_payload = {
            "mission_code": "A3R8",
            "fire_buildings": [],
            "points": [],
            "detection": {}
        }
    except Exception as e:
        print(f"[ERROR] JSON 파일 읽기 실패: {e}, 기본값 사용")
        base_payload = {
            "mission_code": "A3R8",
            "fire_buildings": [],
            "points": [],
            "detection": {}
        }
    
    # mission_code와 fire_buildings는 파일에서 가져온 값 유지
    # points와 detection만 함수 파라미터로 업데이트
    # detection 형식: {point_name: [{type: class_name, count: count}, ...]}
    
    # 기존 detection과 새로운 detection 병합
    # sendjson.ipynb 형식: 각 point별로 배열이 있어야 함 (빈 배열이라도)
    base_detection = base_payload.get("detection", {})
    final_points = points if points is not None else base_payload.get("points", [])
    
    # 모든 point에 대해 빈 배열로 초기화 (sendjson.ipynb 형식 준수)
    merged_detection = {}
    for point_name in final_points:
        # 기존 detection에 있으면 사용, 없으면 빈 배열
        merged_detection[point_name] = base_detection.get(point_name, [])
    
    # 새로운 detection 정보로 업데이트
    if detected_objects is not None:
        for point_name, detection_list in detected_objects.items():
            merged_detection[point_name] = detection_list
    
    final_detection = merged_detection
    
    payload = {
        "mission_code": base_payload.get("mission_code", "A3R8"),
        "fire_buildings": base_payload.get("fire_buildings", []),
        "points": final_points,
        "detection": final_detection
    }

    json_content = json.dumps(payload, indent=2, ensure_ascii=False)
    files = {
        'file': (f"{payload['mission_code']}.json", json_content, 'application/json')
    }

    try:
        print(f"[SEND] dashboard payload → points={payload['points']} detection_keys={list(payload['detection'].keys())}")
        rsp = requests.post(f"{SERVER_URL}/dashboard_json", files=files, timeout=10)
        print("[Server Response]", rsp.text)
        
        # 서버 전송 성공 후 JSON 파일에 payload 업데이트
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
            
            with open(JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"[UPDATE] JSON 파일 업데이트 완료: {JSON_PATH}")
        except Exception as e:
            print(f"[WARNING] JSON 파일 업데이트 실패: {e}")
            
    except Exception as e:
        print("[ERROR] 서버 전송 실패:", e)


def send_dashboard_image(image_path):
    """대시보드로 이미지 파일 전송"""
    image_file = Path(image_path)
    
    if not image_file.exists():
        print(f"[WARNING] 이미지 파일 없음: {image_file}")
        return
    
    try:
        print(f"[SEND IMAGE] 이미지 전송 중: {image_file.name}")
        
        with open(image_file, "rb") as f:
            files = {
                'file': (image_file.name, f, 'image/jpeg')
            }
            
            response = requests.post(
                f'{SERVER_URL}/img/dashboard/fire_building',
                files=files,
                timeout=10
            )
        
        print(f"[SEND IMAGE] 서버 응답: {response.text}")
        
    except Exception as e:
        print(f"[ERROR] 이미지 전송 실패: {e}")

