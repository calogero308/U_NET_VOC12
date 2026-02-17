import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Function that returns the CE loss ignoring the 255 index
def get_loss():
    return nn.CrossEntropyLoss(ignore_index=255)

#Function that returns the optmizer "AdamW"
def get_optimizer(model_params, learning_rate=1e-4, weight_decay=1e-4):
    #AdamW è molto più stabile in segmentation rispetto a Adam
    return optim.AdamW(params=model_params, lr=learning_rate, weight_decay=weight_decay)

#Functions that save the model state -> model parameters, optimizer, epoch of training
def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato: {path}")


#Class definition of Dice Loss 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        # One-hot targets
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes-1), num_classes=num_classes)  # clamp evita valori > num_classes-1
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [N, C, H, W]

        # Softmax sui logits
        probs = F.softmax(logits, dim=1)

        # Crea mask per ignorare i pixel 255
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # [N, 1, H, W]

        # Applica mask a probs e targets
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Flatten
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)

        # Dice
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1 - dice

#Function that calculates the mIoU of the model
def evaluate_model(model, dataloader, device, num_classes=21):
    model.eval()
    
    total_intersection = torch.zeros(num_classes).to(device)
    total_union = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(num_classes):
                pred_inds = (preds == cls)
                target_inds = (masks == cls)

                intersection = (pred_inds & target_inds).sum()
                union = (pred_inds | target_inds).sum()

                total_intersection[cls] += intersection
                total_union[cls] += union

    iou_per_class = total_intersection / (total_union + 1e-6)
    mean_iou = torch.mean(iou_per_class[total_union > 0])

    return mean_iou.item(), iou_per_class.cpu().numpy()

#WORK IN PROGRESS
def pixel_accuracy(output, mask):
    preds = torch.argmax(output, dim=1)
    correct = (preds == mask).float()
    return correct.sum() / correct.numel()


