"""
Upload YOLO Tank Detection models to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Hugging Face settings
REPO_ID = "Hunjun/yolo-tank-detection"  # Change to your username


def upload_models():
    """Upload all models to Hugging Face Hub"""

    api = HfApi()

    # Create repository (if not exists)
    print(f"Creating repository: {REPO_ID}")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"  Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"  Repository might already exist: {e}")

    # Files to upload
    files_to_upload = [
        # Teacher model
        ("models/teacher/yolov8m_tank/weights/best.pt", "teacher/yolov8m_tank_best.pt"),
        ("models/teacher/yolov8m_tank/results.csv", "teacher/results.csv"),

        # Student fine-tuned model
        ("models/student/yolov8n_distilled/weights/best.pt", "student/yolov8n_finetuned_best.pt"),

        # Student KD model
        ("models/student/yolov8n_kd_hinton/weights/best.pt", "student/yolov8n_kd_hinton_best.pt"),
        ("models/student/yolov8n_kd_hinton/results.csv", "student/yolov8n_kd_hinton_results.csv"),
        ("models/student/yolov8n_kd_hinton/evaluation.txt", "student/yolov8n_kd_hinton_evaluation.txt"),

        # Quantized model for web
        ("models/optimized/student_quantized.onnx", "optimized/student_quantized.onnx"),
    ]

    print("\nUploading files...")
    for local_path, hub_path in files_to_upload:
        full_local_path = os.path.join(PROJECT_ROOT, local_path)

        if os.path.exists(full_local_path):
            file_size = os.path.getsize(full_local_path) / (1024 * 1024)
            print(f"  Uploading: {hub_path} ({file_size:.2f} MB)")

            try:
                api.upload_file(
                    path_or_fileobj=full_local_path,
                    path_in_repo=hub_path,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"    Uploaded!")
            except Exception as e:
                print(f"    Error: {e}")
        else:
            print(f"  Skipping (not found): {local_path}")

    # Create model card
    model_card = """---
license: mit
tags:
- yolo
- object-detection
- knowledge-distillation
- quantization
- military
- tank-detection
datasets:
- roboflow/military-object-detection
metrics:
- mAP
---

# YOLO Tank Detection Models

Military object detection models trained with Knowledge Distillation and Post-Training Quantization.

## Models

| Model | mAP50 | mAP50-95 | Size |
|-------|-------|----------|------|
| Teacher (YOLOv8m) | 86.40% | 62.78% | 49.6 MB |
| Student Fine-tuned (YOLOv8n) | 84.61% | 60.64% | 5.96 MB |
| Student KD (YOLOv8n) | 79.28% | 53.38% | ~6 MB |
| Student Quantized (INT8) | 84.31% | 60.50% | 3.20 MB |

## Usage

```python
from huggingface_hub import hf_hub_download

# Download quantized model for web deployment
model_path = hf_hub_download(
    repo_id="Hunjun/yolo-tank-detection",
    filename="optimized/student_quantized.onnx"
)

# Or download teacher model
teacher_path = hf_hub_download(
    repo_id="Hunjun/yolo-tank-detection",
    filename="teacher/yolov8m_tank_best.pt"
)
```

## Classes

- Airplane
- Helicopter
- Person
- Tank
- Vehicle

## Training

- Teacher: YOLOv8m fine-tuned on Military Dataset (20 epochs)
- Student: YOLOv8n fine-tuned / Knowledge Distillation (20 epochs)
- Quantization: INT8 dynamic quantization via ONNX Runtime

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
"""

    # Upload model card
    print("\nUploading model card (README.md)...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("  Model card uploaded!")

    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"View models at: https://huggingface.co/{REPO_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    upload_models()
