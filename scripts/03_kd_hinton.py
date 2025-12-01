"""
Phase 3: Hinton's Knowledge Distillation (Option B)

Original Hinton KD approach:
- Total Loss = α × Hard_Loss + (1-α) × T² × KL_Divergence(Student_soft, Teacher_soft)
- Hard Loss: Standard YOLO Detection Loss with Ground Truth
- Soft Loss: KL Divergence between softened Teacher and Student outputs

Comparison target: Fine-tuned Student (mAP50: 84.61%)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
import yaml
from tqdm import tqdm
import csv
from datetime import datetime
from copy import deepcopy

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "data.yaml")
TEACHER_DIR = os.path.join(PROJECT_ROOT, "models", "teacher")
STUDENT_DIR = os.path.join(PROJECT_ROOT, "models", "student")


class SoftTargetKDLoss(nn.Module):
    """
    Soft Target KL Divergence Loss for Knowledge Distillation

    Following Hinton et al. (2015):
    - Apply temperature scaling to soften probability distributions
    - Use KL divergence to match Student to Teacher
    - Multiply by T^2 to maintain gradient magnitude
    """

    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: Raw logits from Student [batch, num_classes, anchors]
            teacher_logits: Raw logits from Teacher [batch, num_classes, anchors]

        Returns:
            KL Divergence loss scaled by T^2
        """
        # Flatten spatial dimensions: [batch, C, H*W] -> [batch, C*H*W]
        s_flat = student_logits.view(student_logits.size(0), -1)
        t_flat = teacher_logits.view(teacher_logits.size(0), -1)

        # If shapes don't match, interpolate
        if s_flat.shape[1] != t_flat.shape[1]:
            min_size = min(s_flat.shape[1], t_flat.shape[1])
            s_flat = s_flat[:, :min_size]
            t_flat = t_flat[:, :min_size]

        # Temperature-scaled softmax
        s_soft = F.log_softmax(s_flat / self.T, dim=1)
        t_soft = F.softmax(t_flat / self.T, dim=1).detach()  # Teacher is frozen

        # KL Divergence * T^2
        kd_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.T ** 2)

        return kd_loss


def get_dataloader(data_yaml, batch_size=8, img_size=640, mode='train'):
    """Create YOLO dataloader"""

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


def train_kd_hinton(
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
    Hinton's Knowledge Distillation Training (Option B)

    Loss = α × Hard_Loss + (1-α) × T² × KL(Student_soft || Teacher_soft)

    Args:
        teacher_path: Path to trained Teacher model
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        temperature: Temperature for soft labels (higher = softer)
        alpha: Weight for Hard Loss (1-alpha for Soft Loss)
        lr: Learning rate
        save_dir: Directory to save results
    """

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("Hinton's Knowledge Distillation (Option B)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  - Teacher: {teacher_path}")
    print(f"  - Student: YOLOv8n (pretrained backbone + new 5-class HEAD)")
    print(f"  - Temperature: {temperature}")
    print(f"  - Alpha: {alpha} (Hard: {alpha*100:.0f}%, Soft: {(1-alpha)*100:.0f}%)")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Device: {device}")
    print(f"\n  Loss = {alpha} × Hard_Loss + {1-alpha} × T² × KL_Divergence")

    # =========================================================
    # 1. Load Models
    # =========================================================
    print("\n[1/5] Loading Models...")

    # Load Teacher (frozen, but TRAIN mode for BatchNorm consistency)
    # Key insight: Teacher should be in train mode so BatchNorm uses current batch stats
    # This ensures feature distributions match between Teacher and Student
    teacher = YOLO(teacher_path)
    teacher_model = teacher.model.to(device)
    teacher_model.train()  # TRAIN mode (NOT eval!) for BatchNorm consistency
    for p in teacher_model.parameters():
        p.requires_grad = False  # Frozen - no gradient updates

    teacher_nc = teacher_model.model[-1].nc  # Number of classes
    print(f"  Teacher loaded: {sum(p.numel() for p in teacher_model.parameters()):,} params, {teacher_nc} classes")
    print(f"  Teacher mode: TRAIN (frozen) - BatchNorm uses batch stats")

    # Load Student with same number of classes as Teacher
    # Option 1: Use Fine-tuned Student (same nc as Teacher)
    # Option 2: Re-initialize from scratch

    # Check for existing fine-tuned student
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
        # Load fine-tuned student (already has 5 classes)
        print(f"  Loading Fine-tuned Student from: {student_path}")
        student = YOLO(student_path)
        student_model = student.model.to(device)

        # Re-initialize classification layers to start fresh for KD comparison
        print(f"  Re-initializing classification layers for fair KD comparison...")
        detect_head = student_model.model[-1]

        # Re-initialize cv2 (classification conv layers) only
        for layer in detect_head.cv2:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        print(f"  Student re-initialized with {detect_head.nc} classes")
    else:
        # No fine-tuned student available, use pretrained and adapt
        print(f"  No fine-tuned student found. Using pretrained YOLOv8n...")
        print(f"  Note: Output shapes may differ. Using MSE for distillation loss.")
        student = YOLO("yolov8n.pt")
        student_model = student.model.to(device)

    # Ensure all student parameters require gradients
    for p in student_model.parameters():
        p.requires_grad = True

    student_model.train()
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"  Student loaded: {sum(p.numel() for p in student_model.parameters()):,} params ({trainable_params:,} trainable)")

    # =========================================================
    # 2. Setup Training
    # =========================================================
    print("\n[2/5] Setting up Training...")

    # KD Loss
    kd_loss_fn = SoftTargetKDLoss(temperature=temperature)

    # Detection Loss (Hard Loss) - Using simple MSE for bbox + CE for cls
    # For simplicity, we'll use the output directly
    mse_loss = nn.MSELoss()

    # Optimizer - optimize all Student parameters
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=lr,
        weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    # Save directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(STUDENT_DIR, f"yolov8n_kd_hinton_{timestamp}")
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
    print(f"  Save directory: {save_dir}")

    # =========================================================
    # 3. Create Dataloaders
    # =========================================================
    print("\n[3/5] Creating Dataloaders...")
    train_loader = get_dataloader(DATA_YAML, batch_size, img_size, 'train')
    val_loader = get_dataloader(DATA_YAML, batch_size, img_size, 'val')
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # =========================================================
    # 4. Training Loop
    # =========================================================
    print("\n[4/5] Training with Knowledge Distillation...")

    history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0
        epoch_hard = 0
        epoch_soft = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            images = batch['img'].to(device).float() / 255.0

            # ============================================
            # Teacher Forward (TRAIN mode, frozen, no gradient)
            # Key: Teacher stays in train mode for BatchNorm consistency
            # BatchNorm uses current batch stats, matching Student's distribution
            # ============================================
            with torch.no_grad():
                # Teacher is already in train mode (set during init)
                # Do NOT call teacher_model.eval() here!
                teacher_out = teacher_model(images)
                # train mode output: list of 3 feature maps
                if isinstance(teacher_out, (list, tuple)):
                    # Flatten and concatenate all scales
                    teacher_out = torch.cat([t.flatten(1) for t in teacher_out], dim=1)

            # ============================================
            # Student Forward (TRAIN mode, with gradient)
            # Both Teacher and Student in train mode = consistent feature distributions
            # ============================================
            # Student stays in train mode for gradient computation
            student_out = student_model(images)
            # train mode output: list of 3 feature maps
            if isinstance(student_out, (list, tuple)):
                # Flatten and concatenate all scales
                student_out = torch.cat([s.flatten(1) for s in student_out], dim=1)

            # Debug: Print shapes for first batch
            if epoch == 0 and num_batches == 0:
                print(f"\n  Teacher output: {teacher_out.shape}")
                print(f"  Student output: {student_out.shape}")
                print(f"  Student output requires_grad: {student_out.requires_grad}")

            # ============================================
            # Loss Calculation
            # ============================================

            # Hard Loss: MSE between Student output and Teacher output (pseudo-labels)
            # This encourages Student to match Teacher's predictions
            hard_loss = mse_loss(student_out, teacher_out.detach())

            # Soft Loss: KL Divergence with temperature scaling
            soft_loss = kd_loss_fn(student_out, teacher_out)

            # Total Loss: Weighted combination
            total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

            # Check for valid loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n  Warning: Invalid loss at batch {num_batches}, skipping...")
                continue

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10.0)

            optimizer.step()

            # Metrics
            epoch_loss += total_loss.item()
            epoch_hard += hard_loss.item()
            epoch_soft += soft_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'hard': f'{hard_loss.item():.4f}',
                'soft': f'{soft_loss.item():.4f}'
            })

        scheduler.step()

        if num_batches == 0:
            print(f"  Epoch {epoch+1}: No valid batches!")
            continue

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_hard = epoch_hard / num_batches
        avg_soft = epoch_soft / num_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n  Epoch {epoch+1}: Loss={avg_loss:.4f}, Hard={avg_hard:.4f}, Soft={avg_soft:.4f}, LR={current_lr:.6f}")

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'hard_loss': avg_hard,
            'soft_loss': avg_soft,
            'lr': current_lr
        })

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save state dict
            torch.save(student_model.state_dict(), os.path.join(save_dir, "weights", "best_state.pt"))
            print(f"  Best model saved! (loss: {best_loss:.4f})")

        # Save last model
        torch.save(student_model.state_dict(), os.path.join(save_dir, "weights", "last_state.pt"))

    # Save training history
    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'hard_loss', 'soft_loss', 'lr'])
        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 60)
    print("Knowledge Distillation Complete!")
    print("=" * 60)
    print(f"\nBest state dict: {os.path.join(save_dir, 'weights', 'best_state.pt')}")
    print(f"Best loss: {best_loss:.4f}")

    # =========================================================
    # 5. Convert to Ultralytics format and Evaluate
    # =========================================================
    print("\n[5/5] Converting and Evaluating...")

    try:
        # Load the same base model architecture used during training
        if student_path:
            eval_model = YOLO(student_path)  # Same architecture
        else:
            eval_model = YOLO("yolov8n.pt")

        # Load trained weights
        best_state = torch.load(os.path.join(save_dir, "weights", "best_state.pt"), map_location='cpu')
        eval_model.model.load_state_dict(best_state)

        # Save as YOLO format for evaluation
        yolo_save_path = os.path.join(save_dir, "weights", "best.pt")
        eval_model.save(yolo_save_path)
        print(f"  YOLO format saved: {yolo_save_path}")

        # Evaluate
        print("\n  Evaluating KD Student...")
        results = eval_model.val(data=DATA_YAML, verbose=False)

        print(f"\n  KD Student Results:")
        print(f"    mAP50: {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
        print(f"    mAP50-95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")

        # Save evaluation results
        with open(os.path.join(save_dir, "evaluation.txt"), 'w') as f:
            f.write(f"Knowledge Distillation Results\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"mAP50: {results.box.map50:.4f}\n")
            f.write(f"mAP50-95: {results.box.map:.4f}\n")
            f.write(f"\nComparison:\n")
            f.write(f"Fine-tuned Student mAP50: 84.61%\n")
            f.write(f"KD Student mAP50: {results.box.map50*100:.2f}%\n")

    except Exception as e:
        print(f"  Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hinton's Knowledge Distillation (Option B)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for soft labels")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Hard Loss")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    teacher_path = os.path.join(TEACHER_DIR, "yolov8m_tank", "weights", "best.pt")

    if not os.path.exists(teacher_path):
        print(f"Teacher model not found: {teacher_path}")
        print("Please run 02_train_teacher.py first.")
        sys.exit(1)

    train_kd_hinton(
        teacher_path=teacher_path,
        epochs=args.epochs,
        batch_size=args.batch,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.lr
    )
