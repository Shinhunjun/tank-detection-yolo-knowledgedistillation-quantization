"""
Phase 3: Custom Knowledge Distillation (Response-based KD)

Hinton 논문의 원래 Knowledge Distillation 구현:
- Teacher와 Student의 최종 출력(soft labels)만 사용
- 추론 모드 출력 사용으로 동일한 shape 보장: [batch, 9, 8400]
- KL Divergence로 soft label 모방

Loss = α × Hard_Loss + (1-α) × T² × KL(student_soft || teacher_soft)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import yaml
from tqdm import tqdm
import csv
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data.yaml")
TEACHER_DIR = os.path.join(PROJECT_ROOT, "models", "teacher")
STUDENT_DIR = os.path.join(PROJECT_ROOT, "models", "student")


class ResponseKDLoss(nn.Module):
    """
    Response-based Knowledge Distillation Loss (Hinton et al., 2015)

    Teacher와 Student의 최종 출력을 Temperature로 soften하여 비교
    """

    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_out, teacher_out):
        """
        Args:
            student_out: Student flattened 출력 [batch, features]
            teacher_out: Teacher flattened 출력 [batch, features]

        Returns:
            KL Divergence loss with temperature scaling
        """
        # 이미 flatten된 상태로 들어옴
        s_flat = student_out
        t_flat = teacher_out

        # Shape 맞추기 (Teacher가 더 클 수 있음)
        if s_flat.shape[1] != t_flat.shape[1]:
            # Teacher를 Student 크기에 맞게 보간
            t_flat = F.interpolate(
                t_flat.unsqueeze(1),
                size=s_flat.shape[1],
                mode='linear',
                align_corners=False
            ).squeeze(1)

        # Temperature scaling + Softmax
        s_soft = F.log_softmax(s_flat / self.T, dim=1)
        t_soft = F.softmax(t_flat / self.T, dim=1)

        # KL Divergence * T^2 (gradient magnitude 보정)
        loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.T ** 2)

        return loss


def get_dataloader(data_yaml, batch_size=8, img_size=640, mode='train'):
    """Create dataloader"""

    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    if mode == 'train':
        img_path = os.path.join(PROJECT_ROOT, "data", "train", "images")
    else:
        img_path = os.path.join(PROJECT_ROOT, "data", "valid", "images")

    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml
    cfg.imgsz = img_size
    cfg.batch = batch_size

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=img_path,
        batch=batch_size,
        data=data_cfg,
        mode=mode,
        rect=False,
        stride=32
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch=batch_size,
        workers=4,
        shuffle=(mode == 'train'),
        rank=-1
    )

    return dataloader


def train_with_kd(
    teacher_path,
    epochs=20,
    batch_size=8,
    img_size=640,
    temperature=4.0,
    alpha=0.5,
    lr=0.001,
    save_dir=None
):
    """
    Response-based Knowledge Distillation Training

    Hinton 논문 방식: Teacher의 soft label을 Student가 모방
    """

    # Device 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("Response-based Knowledge Distillation (Hinton et al.)")
    print("=" * 60)

    print(f"\n설정:")
    print(f"  - Teacher: {teacher_path}")
    print(f"  - Student: YOLOv8n (pretrained)")
    print(f"  - Temperature: {temperature}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Device: {device}")
    print(f"  - Loss: α×Hard + (1-α)×T²×KL(soft)")

    # =========================================================
    # 1. 모델 로드
    # =========================================================
    print("\n[1/4] 모델 로드 중...")

    # Teacher (frozen, eval mode)
    teacher = YOLO(teacher_path)
    teacher_model = teacher.model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print(f"  ✅ Teacher 로드: {sum(p.numel() for p in teacher_model.parameters()):,} params")

    # Student (trainable) - 같은 클래스 수(5)로 fine-tuned된 모델 사용
    # 기존 fine-tuned student 모델 경로들
    student_paths = [
        os.path.join(STUDENT_DIR, "yolov8n_distilled", "weights", "best.pt"),
        os.path.join(STUDENT_DIR, "yolov8n_tank", "weights", "best.pt"),
    ]

    student_path = None
    for path in student_paths:
        if os.path.exists(path):
            student_path = path
            break

    if student_path:
        print(f"  📦 Fine-tuned Student 로드: {student_path}")
        student = YOLO(student_path)
    else:
        raise FileNotFoundError(
            "Fine-tuned Student 모델이 필요합니다. "
            "먼저 03_distillation.py를 실행하여 Student를 5 classes로 학습하세요."
        )

    student_model = student.model.to(device)
    print(f"  ✅ Student 로드: {sum(p.numel() for p in student_model.parameters()):,} params")
    print(f"  ✅ Student nc: {student_model.model[-1].nc}")

    # =========================================================
    # 2. 학습 설정
    # =========================================================
    print("\n[2/4] 학습 설정...")

    # KD Loss
    kd_loss_fn = ResponseKDLoss(temperature=temperature)

    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    # Save directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(STUDENT_DIR, f"yolov8n_kd_{timestamp}")
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
    print(f"  ✅ 저장 경로: {save_dir}")

    # =========================================================
    # 3. 데이터 로더
    # =========================================================
    print("\n[3/4] 데이터 로더 생성...")
    train_loader = get_dataloader(DATA_YAML, batch_size, img_size, 'train')
    print(f"  ✅ Train 배치: {len(train_loader)}")

    # =========================================================
    # 4. 학습 루프
    # =========================================================
    print("\n[4/4] Knowledge Distillation 학습...")
    print(f"\n  추론 모드 출력 사용: [batch, 9, 8400]")
    print(f"  Teacher와 Student 출력 shape 동일")

    history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0
        epoch_kd_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            images = batch['img'].to(device).float() / 255.0

            # ============================================
            # Teacher forward (학습 모드 출력 사용)
            # ============================================
            # YOLOv8 학습 모드: 3개 스케일의 feature maps 반환
            # 각 feature map을 flatten하여 사용
            with torch.no_grad():
                teacher_model.eval()
                t_out = teacher_model.model(images)  # 내부 model 직접 호출
                # 학습 모드 출력: list of tensors
                if isinstance(t_out, (list, tuple)):
                    # 모든 출력을 concat하여 하나의 tensor로
                    teacher_out = torch.cat([t.flatten(1) for t in t_out], dim=1)
                else:
                    teacher_out = t_out.flatten(1)

            # ============================================
            # Student forward (학습 모드, gradient 유지)
            # ============================================
            student_model.train()
            s_out = student_model.model(images)  # 내부 model 직접 호출
            if isinstance(s_out, (list, tuple)):
                student_out = torch.cat([s.flatten(1) for s in s_out], dim=1)
            else:
                student_out = s_out.flatten(1)

            # Shape 확인 (첫 배치만)
            if epoch == 0 and num_batches == 0:
                print(f"\n  Teacher output shape: {teacher_out.shape}")
                print(f"  Student output shape: {student_out.shape}")

            # ============================================
            # KD Loss 계산
            # ============================================
            kd_loss = kd_loss_fn(student_out, teacher_out)

            # Total loss (현재는 KD loss만, hard loss 추가 가능)
            total_loss = kd_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += total_loss.item()
            epoch_kd_loss += kd_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'kd': f'{kd_loss.item():.4f}'
            })

        scheduler.step()

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_kd = epoch_kd_loss / num_batches

        print(f"\n  Epoch {epoch+1}: Loss={avg_loss:.4f}, KD={avg_kd:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'kd_loss': avg_kd,
            'lr': scheduler.get_last_lr()[0]
        })

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student_model.state_dict(), os.path.join(save_dir, "weights", "best.pt"))
            print(f"  ✅ Best model saved! (loss: {best_loss:.4f})")

        # Save last
        torch.save(student_model.state_dict(), os.path.join(save_dir, "weights", "last.pt"))

    # Save history
    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'kd_loss', 'lr'])
        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 60)
    print("Knowledge Distillation 완료!")
    print("=" * 60)
    print(f"\nBest model: {os.path.join(save_dir, 'weights', 'best.pt')}")
    print(f"Best loss: {best_loss:.4f}")

    # Convert to Ultralytics format for evaluation
    print("\n모델 변환 중...")
    try:
        student_eval = YOLO("yolov8n.pt")
        student_eval.model.load_state_dict(
            torch.load(os.path.join(save_dir, "weights", "best.pt"), map_location=device)
        )
        # Save as YOLO format
        save_path = os.path.join(save_dir, "weights", "best_yolo.pt")
        torch.save({
            'model': student_eval.model,
            'train_args': {'imgsz': img_size, 'nc': 5}
        }, save_path)
        print(f"  ✅ YOLO 형식 저장: {save_path}")
    except Exception as e:
        print(f"  ⚠️ 변환 실패: {e}")

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Response-based Knowledge Distillation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    teacher_path = os.path.join(TEACHER_DIR, "yolov8m_tank", "weights", "best.pt")

    if not os.path.exists(teacher_path):
        print(f"❌ Teacher 모델이 없습니다: {teacher_path}")
        exit(1)

    train_with_kd(
        teacher_path=teacher_path,
        epochs=args.epochs,
        batch_size=args.batch,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.lr
    )
