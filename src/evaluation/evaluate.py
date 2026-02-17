import torch

from src.models.Unet import U_Net
from src.utils.dataloaders import get_Train_Val_loader_split
from src.utils.train_utils import evaluate_model

import os


def evaluation():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    NUM_CLASSES = 21
    BATCH_SIZE = 8
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(base_dir, "metrics", "unet_checkpoint50.pth")

    #Dataloader (only validation)
    _, val_loader = get_Train_Val_loader_split(BATCH_SIZE=BATCH_SIZE)

    #Model
    model = U_Net(NUM_CLASSES).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean_IoU, IoU_for_class = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=NUM_CLASSES
    )

    print("\n===== EVALUATION RESULTS =====")
    print(f"Mean IoU: {mean_IoU:.4f}\n")

    for cls, iou in enumerate(IoU_for_class):
        print(f"Class {cls:2d} IoU: {iou:.4f}")

if __name__ == "__main__":
    evaluation()