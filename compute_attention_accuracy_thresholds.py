import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils import load_config, get_device
from src.data_preprocessing_cnn import BIWIImageDataset
from src.fusion_variants import create_fusion_variant


def main():
    config = load_config()
    device = get_device(config)

    # Attention fusion, EfficientNet-B0, 1404:99 (best run)
    model_path = Path(
        "models/ablation/dimension_attention/attention_efficientnet_b0_best_1404:99.pth"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = create_fusion_variant(
        fusion_type="attention",
        backbone="efficientnet_b0",
        pretrained=False,
        output_dim=3,
        face_dim=1404,
        pose_dim=99
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_mean = checkpoint["y_mean"].to(device)
    y_std = checkpoint["y_std"].to(device)

    test = np.load("data/splits/biwi_cnn_test.npz", allow_pickle=True)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = BIWIImageDataset(
        test["image_paths"],
        test["labels"],
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_max_err = []

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            preds = model(images)
            preds = preds * y_std + y_mean
            angles = angles * y_std + y_mean

            err = (preds - angles).abs()
            max_err = err.max(dim=1).values
            all_max_err.append(max_err.cpu())

    all_max_err = torch.cat(all_max_err)

    for t in [1, 2, 5]:
        acc = (all_max_err <= t).float().mean().item() * 100
        print(f"Acc@{t}°: {acc:.2f}%")


if __name__ == "__main__":
    main()
