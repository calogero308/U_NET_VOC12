def testing():

    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.models.Unet import U_Net

    # Trasformazione per predizione
    transform_pred = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # Carica immagine
    img_path = "data/archive/VOC2012_test/VOC2012_test/JPEGImages/2008_000014.jpg"
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Applica trasformazione
    transformed = transform_pred(image=img_np)
    img_t = transformed['image'].unsqueeze(0).to("cuda")

    # Carica modello
    model = U_Net(21).to("cuda")
    checkpoint = torch.load("unet_checkpoint50.pth", map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predizione
    with torch.no_grad():
        img_model = model(img_t)
    pred = torch.argmax(img_model, dim=1)  # [1, H, W]

    # ==============================
    # ANALISI DELLE CLASSI PREDOTTE
    # ==============================

    voc_classes = [
        "background","aeroplane","bicycle","bird","boat","bottle","bus",
        "car","cat","chair","cow","diningtable","dog","horse","motorbike",
        "person","pottedplant","sheep","sofa","train","tvmonitor"
    ]

    pred_cpu = pred.squeeze(0).cpu()

    unique_classes = torch.unique(pred_cpu)
    total_pixels = pred_cpu.numel()

    print("\n===== CLASSI PREDOTTE =====")

    for cls in unique_classes:
        cls_int = int(cls.item())
        pixel_count = torch.sum(pred_cpu == cls).item()
        percentage = (pixel_count / total_pixels) * 100

        print(f"Classe {cls_int:2d} ({voc_classes[cls_int]:12s}) "
            f"- Pixel: {pixel_count:6d} "
            f"- {percentage:6.2f}%")


    # Denormalizza per visualizzare immagine originale
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img_show = img_t.squeeze(0).cpu() * std + mean
    img_show = torch.clamp(img_show, 0, 1)

    # Visualizzazione
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].imshow(img_show.permute(1,2,0))
    axes[0].set_title("Immagine originale")
    axes[0].axis("off")
    axes[1].imshow(pred.squeeze(0).cpu(), cmap="jet")
    axes[1].set_title("Predizione")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
