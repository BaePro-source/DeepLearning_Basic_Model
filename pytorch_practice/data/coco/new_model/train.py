"""
Training Script for DeepLabV3+ on COCO Semantic Segmentation
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

from model import DeepLabV3Plus
from dataset import COCOSemSeg
from losses import FocalLoss, DiceLoss
from utils import (
    compute_class_weights,
    evaluate,
    train_one_epoch,
    save_predictions
)
import matplotlib.pyplot as plt

@torch.no_grad()
def compute_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(total, 1)

def plot_training_curves(history, model_name="DeepLabV3Plus"):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.plot(epochs, history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    plt.savefig(f"{model_name}_loss_curve.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"])
    plt.plot(epochs, history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    plt.savefig(f"{model_name}_acc_curve.png", dpi=300)
    plt.close()

def main():
    # ===== Configuration =====
    # Update these paths to your COCO dataset
    coco_root = Path(__file__).resolve().parent.parent
    train_img = coco_root / "images" / "test" / "train2017"
    val_img = coco_root / "images" / "val" / "val2017"
    train_ann = coco_root / "annotations" / "stuff_train2017.json"
    val_ann = coco_root / "annotations" / "stuff_val2017.json"

    # Hyperparameters
    RESIZE = 520
    CROP = 512
    BATCH_SIZE = 6
    NUM_WORKERS = 4
    LR = 1e-4
    EPOCHS = 50
    OUTPUT_STRIDE = 16  # 8 or 16
    
    # Data subset sizes (increase for full training)
    N_TRAIN = 10000
    N_VAL = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===== Dataset =====
    base_train_ds = COCOSemSeg(train_img, train_ann, resize=RESIZE, crop=CROP, train=True)
    base_val_ds = COCOSemSeg(val_img, val_ann, resize=RESIZE, crop=CROP, train=False)

    # Create subsets
    train_ds = Subset(base_train_ds, list(range(min(N_TRAIN, len(base_train_ds)))))
    val_ds = Subset(base_val_ds, list(range(min(N_VAL, len(base_val_ds)))))

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(base_train_ds.cat_ids)
    print(f"Number of classes (including background): {NUM_CLASSES}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=False  # Don't drop last batch
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    # ===== Model =====
    model = DeepLabV3Plus(
        num_classes=NUM_CLASSES, 
        pretrained=True, 
        output_stride=OUTPUT_STRIDE
    ).to(device)

    # ===== Loss Function =====
    # Compute class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)
    
    # Use Focal Loss with class weights
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    
    # Optional: Combine with Dice Loss
    # dice_loss = DiceLoss(num_classes=NUM_CLASSES, exclude_bg=False)
    # You can combine them in train_one_epoch if needed

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ===== Quick sanity check =====
    print("\nSanity check:")
    x0, y0 = next(iter(train_loader))
    print(f"Input shape: {x0.shape}, Target shape: {y0.shape}")
    with torch.no_grad():
        out0 = model(x0.to(device))
    print(f"Output shape: {out0.shape}, Prediction shape: {out0.argmax(1).shape}")

    # ===== Save initial predictions =====
    print("\nSaving initial predictions...")
    save_predictions(model, val_loader, device, NUM_CLASSES, out_dir="vis/before", max_images=4)

    # ===== Training Loop =====
    best_val_miou = -1.0
    
    print(f"\n{'='*50}")
    print(f"Starting training for {EPOCHS} epochs")
    print(f"{'='*50}\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        trian_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = compute_loss(model, val_loader, criterion, device)

        # Evaluate on train set
        train_acc, train_miou_all, train_miou_present = evaluate(
            model, train_loader, device, num_classes=NUM_CLASSES
        )

        # Evaluate on validation set
        val_acc, val_miou_all, val_miou_present = evaluate(
            model, val_loader, device, num_classes=NUM_CLASSES,
            print_topk=10 if (epoch % 10 == 0) else 0
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print results
        print(f"[Epoch {epoch}/{EPOCHS}] loss={loss:.4f}")
        print(
            f"  Train: acc={train_acc*100:.2f}%  "
            f"mIoU(all)={train_miou_all*100:.2f}%  "
            f"mIoU(present)={train_miou_present*100:.2f}%"
        )
        print(
            f"  Val:   acc={val_acc*100:.2f}%  "
            f"mIoU(all)={val_miou_all*100:.2f}%  "
            f"mIoU(present)={val_miou_present*100:.2f}%"
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }


        # Save best model
        if val_miou_present > best_val_miou:
            best_val_miou = val_miou_present
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  [BEST MODEL SAVED] val-mIoU(present)={val_miou_present*100:.2f}%")
        
            save_predictions(
                model, val_loader, device, NUM_CLASSES,
                out_dir="vis/best", max_images=4
            )
        
        print()

    # ===== Final Results =====
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best validation mIoU (present): {best_val_miou*100:.2f}%")
    print(f"{'='*50}\n")

    # Load best model and save predictions
    print("Loading best model and saving predictions...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    
    save_predictions(
        model, val_loader, device, NUM_CLASSES,
        out_dir="vis/best", max_images=4
    )

    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print("\nModels saved:")
    print("  - best_model.pth (best validation mIoU)")
    print("  - final_model.pth (last epoch)")

    plot_training_curves(history, "DeepLabV3Plus")



if __name__ == "__main__":
    main()

    