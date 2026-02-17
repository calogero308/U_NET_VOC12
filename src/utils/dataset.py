from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

#Class that define the Dataset VOC12
class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, has_masks=True):
        self.image_dir = os.path.join(root_dir, "JPEGImages")       #Path of the images
        self.mask_dir = os.path.join(root_dir, "SegmentationClass") #Path of the masks
        self.transform = transform
        self.has_masks = has_masks

        #Check if the image has the relative mask if not skip
        if self.has_masks:
            self.images = sorted([
                img_name for img_name in os.listdir(self.image_dir)
                if os.path.exists(os.path.join(self.mask_dir, img_name.replace(".jpg", ".png")))
            ])
        else:
            # per test senza maschere
            self.images = sorted(os.listdir(self.image_dir))

    #Function that returns the lenght of the dataset
    def __len__(self):
        return len(self.images)

    #Function used to get items in the dataset
    def __getitem__(self, idx):
        img_name = self.images[idx]

        # --- Image ---
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.has_masks:
            # --- Mask ---
            mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
            mask = np.array(Image.open(mask_path))

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].long()

            return image, mask
        else:
            if self.transform:
                image = self.transform(image=image)["image"]
            return image