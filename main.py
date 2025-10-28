import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ----------------------
# Dataset
# ----------------------
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = []
        for city in os.listdir(img_dir):
            for fname in os.listdir(os.path.join(img_dir, city)):
                self.images.append((os.path.join(img_dir, city, fname),
                                    os.path.join(mask_dir, city, fname.replace("leftImg8bit", "gtFine_labelIds"))))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, mask_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.long()


def main():
    # ----------------------
    # Transforms
    # ----------------------
    train_transform = A.Compose([
        A.Resize(256, 512),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 512),
        ToTensorV2(),
    ])

    # ----------------------
    # Paths
    # ----------------------
    train_img_dir = os.path.join(os.getcwd(), "data", "leftImg8bit_trainvaltest", "leftImg8bit", "train")
    train_mask_dir = os.path.join(os.getcwd(), "data", "gtFine_trainvaltest", "gtFine", "train")
    val_img_dir = os.path.join(os.getcwd(), "data", "leftImg8bit_trainvaltest", "leftImg8bit", "val")
    val_mask_dir = os.path.join(os.getcwd(), "data", "gtFine_trainvaltest", "gtFine", "val")

    train_ds = CityscapesDataset(train_img_dir, train_mask_dir, train_transform)
    val_ds = CityscapesDataset(val_img_dir, val_mask_dir, val_transform)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    # ----------------------
    # Model
    # ----------------------
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = smp.Unet(
        encoder_name="resnet18",
        classes=34,
        encoder_weights="imagenet",
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='multiclass')

    # ----------------------
    # Training Loop
    # ----------------------
    EPOCHS = 1
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(device).float(), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            
            if i > 30: # TODO: only for debug speedup, remove for real run
                break
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    # ----------------------
    # Validation/Testing Loop
    # ----------------------
    model.eval()
    val_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device).float(), masks.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            
            # Optional: compute pixel accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
            
            if i > 30: # TODO: only for debug speedup, remove for real run
                break
    
    avg_val_loss = val_loss / len(val_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    print(f"Validation: Average Loss = {avg_val_loss:.4f}, Pixel Accuracy = {pixel_accuracy:.3%}")

    # ----------------------
    # Inference on one image
    # ----------------------
    model.eval()
    with torch.no_grad():
        sample_img, gt_mask = val_ds[0]
        x = sample_img.unsqueeze(0).to(device).float()
        pred = torch.argmax(model(x), dim=1).squeeze().cpu().numpy()

    # Convert tensors for plotting
    img_np = sample_img.permute(1, 2, 0).cpu().numpy()
    gt_mask_np = gt_mask.cpu().numpy()

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(img_np)
    plt.title("Input / Ground Truth Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gt_mask_np)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred)
    plt.title("Model Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
