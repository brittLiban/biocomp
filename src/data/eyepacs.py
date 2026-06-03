from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DR_GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EyePACSDataset(Dataset):
    """EyePACS diabetic retinopathy dataset.

    Expects:
        image_dir: directory containing .jpeg images
        labels_csv: path to trainLabels.csv (columns: image, level)
    """

    def __init__(self, image_dir: str, labels_csv: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform or DEFAULT_TRANSFORM
        df = pd.read_csv(labels_csv)
        # Drop rows where image file doesn't exist
        df["path"] = df["image"].apply(lambda x: self.image_dir / f"{x}.jpeg")
        df = df[df["path"].apply(lambda p: p.exists())].reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["level"]), str(row["image"])

    def __repr__(self):
        return f"EyePACSDataset(n={len(self)}, image_dir={self.image_dir})"
