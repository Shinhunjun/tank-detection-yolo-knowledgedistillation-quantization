"""
Phase 1: Download YOLO Military Dataset from Roboflow
Dataset: https://universe.roboflow.com/rl4pcd/yolo-military-s48o9
"""

import os
from roboflow import Roboflow

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def download_dataset():
    """
    Roboflow에서 YOLO Military Dataset 다운로드

    Note: Roboflow API Key가 필요합니다.
    1. https://roboflow.com 에서 계정 생성
    2. Settings > API Key에서 키 확인
    3. 환경변수 ROBOFLOW_API_KEY 설정 또는 아래에 직접 입력
    """

    # API Key 설정 (환경변수 또는 직접 입력)
    api_key = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

    if api_key == "YOUR_API_KEY_HERE":
        print("=" * 60)
        print("Roboflow API Key가 필요합니다!")
        print()
        print("방법 1: 환경변수 설정")
        print("  export ROBOFLOW_API_KEY='your_api_key'")
        print()
        print("방법 2: 이 파일에서 api_key 직접 수정")
        print()
        print("API Key 얻는 방법:")
        print("  1. https://roboflow.com 접속")
        print("  2. 계정 생성/로그인")
        print("  3. Settings > API Key 확인")
        print("=" * 60)
        return

    print("Roboflow 연결 중...")
    rf = Roboflow(api_key=api_key)

    print("YOLO Military Dataset 다운로드 중...")
    project = rf.workspace("rl4pcd").project("yolo-military-s48o9")

    # YOLOv8 포맷으로 다운로드
    dataset = project.version(1).download("yolov8", location=DATA_DIR)

    print(f"\n다운로드 완료!")
    print(f"위치: {DATA_DIR}")
    print(f"\n데이터셋 정보:")
    print(f"  - 클래스: Airplane, Helicopter, Person, Tank, Vehicle")
    print(f"  - 형식: YOLOv8")


def check_dataset():
    """다운로드된 데이터셋 확인"""

    yaml_path = os.path.join(DATA_DIR, "data.yaml")

    if os.path.exists(yaml_path):
        print("\n데이터셋이 이미 존재합니다.")
        print(f"위치: {yaml_path}")

        with open(yaml_path, 'r') as f:
            print("\ndata.yaml 내용:")
            print(f.read())
        return True

    return False


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO Military Dataset 다운로드")
    print("=" * 60)

    if not check_dataset():
        download_dataset()
    else:
        print("\n이미 데이터가 있습니다. 재다운로드하려면 data/ 폴더를 삭제하세요.")
