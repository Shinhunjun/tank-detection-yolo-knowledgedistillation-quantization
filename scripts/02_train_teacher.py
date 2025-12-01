"""
Phase 2: Train Teacher Model (YOLOv8m)

Teacher 모델을 전차 데이터셋으로 Fine-tuning합니다.
Mac M4 MPS 백엔드를 사용하여 학습합니다.
"""

import os
import torch
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data.yaml")
TEACHER_DIR = os.path.join(PROJECT_ROOT, "models", "teacher")


def check_device():
    """사용 가능한 디바이스 확인"""
    print("=" * 60)
    print("디바이스 정보")
    print("=" * 60)

    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) 사용 가능")
        device = "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA (NVIDIA GPU) 사용 가능")
        device = "cuda"
    else:
        print("⚠️ GPU 사용 불가, CPU 사용")
        device = "cpu"

    print(f"선택된 디바이스: {device}")
    return device


def train_teacher(epochs=100, batch_size=16, img_size=640):
    """
    Teacher 모델 학습 (YOLOv8m)

    Args:
        epochs: 학습 에포크 수
        batch_size: 배치 크기 (M4 메모리에 맞게 조정)
        img_size: 입력 이미지 크기
    """

    device = check_device()

    # 데이터셋 확인
    if not os.path.exists(DATA_YAML):
        print(f"\n❌ 데이터셋을 찾을 수 없습니다: {DATA_YAML}")
        print("먼저 01_download_data.py를 실행하세요.")
        return None

    print("\n" + "=" * 60)
    print("Teacher 모델 학습 시작 (YOLOv8m)")
    print("=" * 60)

    # YOLOv8m 모델 로드 (COCO pretrained)
    print("\nYOLOv8m pretrained 모델 로드 중...")
    teacher = YOLO("yolov8m.pt")

    # 학습 설정
    print(f"\n학습 설정:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {img_size}")
    print(f"  - Device: {device}")
    print(f"  - Data: {DATA_YAML}")

    # 학습 시작
    print("\n학습 시작...")
    results = teacher.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=img_size,
        device=device,
        batch=batch_size,
        patience=20,          # Early stopping
        save=True,
        project=TEACHER_DIR,
        name="yolov8m_tank",
        exist_ok=True,
        plots=True,           # 학습 그래프 저장
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Teacher 모델 학습 완료!")
    print("=" * 60)

    # 결과 경로
    best_model_path = os.path.join(TEACHER_DIR, "yolov8m_tank", "weights", "best.pt")
    print(f"\n최고 성능 모델: {best_model_path}")

    return best_model_path


def evaluate_teacher(model_path=None):
    """학습된 Teacher 모델 평가"""

    if model_path is None:
        model_path = os.path.join(TEACHER_DIR, "yolov8m_tank", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return

    print("\n" + "=" * 60)
    print("Teacher 모델 평가")
    print("=" * 60)

    model = YOLO(model_path)

    # Validation set으로 평가
    results = model.val(data=DATA_YAML)

    print(f"\n평가 결과:")
    print(f"  - mAP50: {results.box.map50:.4f}")
    print(f"  - mAP50-95: {results.box.map:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8m Teacher Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing model")

    args = parser.parse_args()

    if args.eval_only:
        evaluate_teacher()
    else:
        best_path = train_teacher(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size
        )

        if best_path:
            evaluate_teacher(best_path)
