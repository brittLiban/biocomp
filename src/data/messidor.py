from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MessidorDataset(Dataset):
    """Messidor-2 diabetic retinopathy dataset.

    Expects:
        image_dir: directory containing images (.png or .jpg)
        labels_csv: path to MESSIDOR_Base*.csv or MESSIDOR-2 label file
                    (must have columns: image_id/Filename and adjudicated_dr_grade or similar)
    """

    def __init__(self, image_dir: str, labels_csv: str, transform=None,
                 image_col: str = "image_id", label_col: str = "adjudicated_dr_grade"):
        self.image_dir = Path(image_dir)
        self.transform = transform or DEFAULT_TRANSFORM
        df = pd.read_csv(labels_csv)
        df = df.rename(columns={image_col: "image", label_col: "level"})
        # Find image files (png or jpg)
        def find_image(name):
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                p = self.image_dir / f"{name}{ext}"
                if p.exists():
                    return p
            return None
        df["path"] = df["image"].apply(find_image)
        df = df[df["path"].notna()].reset_index(drop=True)
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
        return f"MessidorDataset(n={len(self)}, image_dir={self.image_dir})"
