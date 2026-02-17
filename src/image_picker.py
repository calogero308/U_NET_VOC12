import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as Ts
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Pick an image form the test dataset
img_path = "data/archive/VOC2012_test/VOC2012_test/JPEGImages/2008_000001.jpg"
img = Image.open(img_path)  #Open the path

#Get the original size
width, height = img.size
print(f"Originale: width={width}px, height={height}px")


#BLOCCO CHE VA A MODIFICARE L'IMMAGINE IN MODO DA POTER ESSERE PASSATA ALL'ENCODER
#---------------------------------------------------------------------------------
#Simple transformation of the image used for testing mode
transform = Ts.Compose([
    Ts.Resize((256,256)),
    Ts.ToTensor()
])

#Transformation using random augmentation 
train_transform = A.Compose([
    A.RandomResizedCrop(256, 256, scale=(0.5, 1.5)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    
    ToTensorV2()
])

#Appling the tranformation to the immage
img_t = train_transform(img)

#Get the tensor shape
C, H, W = img_t.shape
print(f"Trasformata: canali={C}, height={H}px, width={W}px")
#---------------------------------------------------------------------------------

#Plotting the immage on the screen
plt.imshow(img_t.permute(1, 2, 0))  # CHW -> HWC
plt.axis("off")
plt.show()
