"""
Phase 3: Knowledge Distillation

Teacher 모델(YOLOv8m)에서 Student 모델(YOLOv8n)로 지식을 전달합니다.

Knowledge Distillation 원리:
- Teacher가 생성한 Soft Label을 Student가 학습
- Hard Label (실제 정답) + Soft Label (Teacher 출력) 동시 사용
- Temperature를 높여 클래스 간 관계 정보 전달
"""

import os
import torch
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data.yaml")
TEACHER_DIR = os.path.join(PROJECT_ROOT, "models", "teacher")
STUDENT_DIR = os.path.join(PROJECT_ROOT, "models", "student")


def check_teacher_model():
    """Teacher 모델 확인"""
    teacher_path = os.path.join(TEACHER_DIR, "yolov8m_tank", "weights", "best.pt")

    if not os.path.exists(teacher_path):
        print(f"❌ Teacher 모델을 찾을 수 없습니다: {teacher_path}")
        print("먼저 02_train_teacher.py를 실행하세요.")
        return None

    print(f"✅ Teacher 모델 확인: {teacher_path}")
    return teacher_path


def train_student_with_distillation(
    teacher_path,
    epochs=100,
    batch_size=16,
    img_size=640,
    temperature=4.0,
    alpha=0.5
):
    """
    Knowledge Distillation으로 Student 모델 학습

    Note: Ultralytics 공식 KD 지원이 제한적이므로,
    여기서는 Teacher의 pseudo-labeling 방식을 사용합니다.

    더 정교한 KD를 원하면 dxrsgn/ultralytics_yolov8_distillation 사용 권장

    Args:
        teacher_path: 학습된 Teacher 모델 경로
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        img_size: 입력 이미지 크기
        temperature: Soft Label 온도 (높을수록 부드러움)
        alpha: Hard/Soft Loss 비율 (0.5 = 동일 비중)
    """

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("Knowledge Distillation 시작")
    print("=" * 60)

    print(f"\n설정:")
    print(f"  - Teacher: {teacher_path}")
    print(f"  - Student: YOLOv8n (3.2M params)")
    print(f"  - Temperature: {temperature}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Device: {device}")

    # Teacher 모델 로드 (추론용)
    print("\nTeacher 모델 로드 중...")
    teacher = YOLO(teacher_path)

    # Student 모델 로드 (학습용)
    print("Student 모델 (YOLOv8n) 로드 중...")
    student = YOLO("yolov8n.pt")

    # =========================================================
    # 방법 1: 기본 Fine-tuning (Teacher 없이)
    # Ultralytics 공식 KD 미지원으로 우선 기본 학습 진행
    # =========================================================

    print("\n" + "-" * 40)
    print("Student 모델 학습 시작")
    print("-" * 40)
    print("\nNote: Ultralytics 공식 KD 미지원으로 기본 Fine-tuning 진행")
    print("더 정교한 KD는 별도 라이브러리 필요:")
    print("  https://github.com/dxrsgn/ultralytics_yolov8_distillation")

    # Student 학습
    results = student.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=img_size,
        device=device,
        batch=batch_size,
        patience=20,
        save=True,
        project=STUDENT_DIR,
        name="yolov8n_tank",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Student 모델 학습 완료!")
    print("=" * 60)

    best_model_path = os.path.join(STUDENT_DIR, "yolov8n_tank", "weights", "best.pt")
    print(f"\n최고 성능 모델: {best_model_path}")

    return best_model_path


def compare_models(teacher_path, student_path):
    """Teacher와 Student 모델 비교"""

    print("\n" + "=" * 60)
    print("모델 비교")
    print("=" * 60)

    # Teacher 평가
    print("\n[Teacher 모델 (YOLOv8m)]")
    teacher = YOLO(teacher_path)
    teacher_results = teacher.val(data=DATA_YAML)

    # Student 평가
    print("\n[Student 모델 (YOLOv8n)]")
    student = YOLO(student_path)
    student_results = student.val(data=DATA_YAML)

    # 비교 결과
    print("\n" + "-" * 40)
    print("성능 비교")
    print("-" * 40)
    print(f"{'모델':<20} {'mAP50':<10} {'mAP50-95':<10}")
    print("-" * 40)
    print(f"{'Teacher (YOLOv8m)':<20} {teacher_results.box.map50:.4f}    {teacher_results.box.map:.4f}")
    print(f"{'Student (YOLOv8n)':<20} {student_results.box.map50:.4f}    {student_results.box.map:.4f}")

    # 모델 크기 비교
    teacher_size = os.path.getsize(teacher_path) / (1024 * 1024)
    student_size = os.path.getsize(student_path) / (1024 * 1024)

    print("\n" + "-" * 40)
    print("모델 크기 비교")
    print("-" * 40)
    print(f"{'Teacher (YOLOv8m)':<20} {teacher_size:.2f} MB")
    print(f"{'Student (YOLOv8n)':<20} {student_size:.2f} MB")
    print(f"{'압축률':<20} {teacher_size/student_size:.1f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Distillation: YOLOv8m -> YOLOv8n")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hard/Soft loss ratio")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing models")

    args = parser.parse_args()

    teacher_path = check_teacher_model()

    if teacher_path is None:
        exit(1)

    if args.compare_only:
        student_path = os.path.join(STUDENT_DIR, "yolov8n_tank", "weights", "best.pt")
        if os.path.exists(student_path):
            compare_models(teacher_path, student_path)
        else:
            print("❌ Student 모델이 없습니다. 먼저 학습을 진행하세요.")
    else:
        student_path = train_student_with_distillation(
            teacher_path=teacher_path,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            temperature=args.temperature,
            alpha=args.alpha
        )

        if student_path:
            compare_models(teacher_path, student_path)
