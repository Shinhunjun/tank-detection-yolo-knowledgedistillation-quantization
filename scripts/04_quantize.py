"""
Phase 4: Post-Training Quantization (PTQ)

Student 모델을 ONNX로 내보내고 INT8 양자화를 적용합니다.
- 모델 크기 ~4배 감소
- 추론 속도 ~2배 향상
"""

import os
import shutil
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_DIR = os.path.join(PROJECT_ROOT, "models", "student")
OPTIMIZED_DIR = os.path.join(PROJECT_ROOT, "models", "optimized")
WEB_MODELS_DIR = os.path.join(PROJECT_ROOT, "web", "models")


def export_to_onnx(model_path):
    """
    YOLO 모델을 ONNX 형식으로 내보내기

    Args:
        model_path: 학습된 Student 모델 경로

    Returns:
        ONNX 모델 경로
    """

    print("\n" + "=" * 60)
    print("ONNX Export")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        return None

    print(f"모델 로드: {model_path}")
    model = YOLO(model_path)

    # ONNX로 내보내기
    print("\nONNX 형식으로 내보내기...")
    onnx_path = model.export(
        format="onnx",
        dynamic=False,      # 웹 배포를 위해 고정 크기 사용
        simplify=True,      # 모델 단순화
        opset=12,           # ONNX opset 버전
        imgsz=640           # 입력 이미지 크기 고정
    )

    print(f"✅ ONNX 모델 생성: {onnx_path}")

    return onnx_path


def quantize_onnx(onnx_path):
    """
    ONNX 모델 양자화 (INT8)

    Args:
        onnx_path: ONNX 모델 경로

    Returns:
        양자화된 모델 경로
    """

    print("\n" + "=" * 60)
    print("ONNX Quantization (INT8)")
    print("=" * 60)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("❌ onnxruntime 설치 필요: pip install onnxruntime")
        return None

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX 모델을 찾을 수 없습니다: {onnx_path}")
        return None

    # 출력 경로
    os.makedirs(OPTIMIZED_DIR, exist_ok=True)
    quantized_path = os.path.join(OPTIMIZED_DIR, "student_quantized.onnx")

    print(f"입력 모델: {onnx_path}")
    print(f"출력 모델: {quantized_path}")

    # 동적 양자화 (INT8)
    print("\nINT8 동적 양자화 적용 중...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8
    )

    print(f"✅ 양자화 완료: {quantized_path}")

    return quantized_path


def compare_model_sizes(original_path, onnx_path, quantized_path):
    """모델 크기 비교"""

    print("\n" + "=" * 60)
    print("모델 크기 비교")
    print("=" * 60)

    sizes = {}

    if original_path and os.path.exists(original_path):
        sizes["Original (PyTorch)"] = os.path.getsize(original_path)

    if onnx_path and os.path.exists(onnx_path):
        sizes["ONNX (FP32)"] = os.path.getsize(onnx_path)

    if quantized_path and os.path.exists(quantized_path):
        sizes["ONNX Quantized (INT8)"] = os.path.getsize(quantized_path)

    print(f"\n{'모델':<25} {'크기':<15} {'압축률':<10}")
    print("-" * 50)

    base_size = list(sizes.values())[0] if sizes else 1

    for name, size in sizes.items():
        size_mb = size / (1024 * 1024)
        ratio = base_size / size if size > 0 else 0
        print(f"{name:<25} {size_mb:.2f} MB       {ratio:.1f}x")


def copy_to_web_folder(quantized_path):
    """웹 배포 폴더로 모델 복사"""

    if not os.path.exists(quantized_path):
        print("❌ 양자화된 모델이 없습니다.")
        return

    os.makedirs(WEB_MODELS_DIR, exist_ok=True)
    dest_path = os.path.join(WEB_MODELS_DIR, "student_quantized.onnx")

    shutil.copy2(quantized_path, dest_path)
    print(f"\n✅ 웹 배포 폴더로 복사: {dest_path}")


def run_quantization_pipeline():
    """전체 양자화 파이프라인 실행"""

    # Student 모델 경로 (distillation 또는 fine-tuning 결과)
    student_path = os.path.join(STUDENT_DIR, "yolov8n_distilled", "weights", "best.pt")

    # fallback: yolov8n_tank 폴더 확인
    if not os.path.exists(student_path):
        student_path = os.path.join(STUDENT_DIR, "yolov8n_tank", "weights", "best.pt")

    if not os.path.exists(student_path):
        print(f"❌ Student 모델을 찾을 수 없습니다: {student_path}")
        print("먼저 03_distillation.py를 실행하세요.")
        return

    print("=" * 60)
    print("Post-Training Quantization Pipeline")
    print("=" * 60)

    # Step 1: ONNX Export
    onnx_path = export_to_onnx(student_path)
    if onnx_path is None:
        return

    # Step 2: Quantization
    quantized_path = quantize_onnx(onnx_path)
    if quantized_path is None:
        return

    # Step 3: 크기 비교
    compare_model_sizes(student_path, onnx_path, quantized_path)

    # Step 4: 웹 폴더로 복사
    copy_to_web_folder(quantized_path)

    print("\n" + "=" * 60)
    print("양자화 완료!")
    print("=" * 60)
    print(f"\n최종 모델: {quantized_path}")
    print("\n다음 단계: 05_web_deploy.py 실행")


if __name__ == "__main__":
    run_quantization_pipeline()
