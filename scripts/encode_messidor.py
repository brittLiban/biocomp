"""
Encode all Messidor images with RETFound, saving embeddings + filename index.

Saves:
  data/processed/embeddings/messidor_retfound.npy   — (N, 1024) float32
  data/processed/embeddings/messidor_filenames.npy  — (N,) str, stems only (no ext)

Stems are used for joining to labels, which may use different extensions.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from tqdm import tqdm

IMAGE_DIR = Path("C:/Users/liban/Documents/biocomp/data/raw/messidor/images/IMAGES")
OUT_DIR   = Path("C:/Users/liban/Documents/biocomp/data/processed/embeddings")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageFolderWithNames(Dataset):
    def __init__(self, image_dir: Path, transform):
        self.paths = sorted(image_dir.glob("*"))
        self.paths = [p for p in self.paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), p.stem  # stem = no extension


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    import sys
    sys.path.insert(0, "C:/Users/liban/Documents/biocomp")
    from src.encoders.retfound import load_retfound

    model = load_retfound(device=device)
    print("RETFound loaded.")

    dataset = ImageFolderWithNames(IMAGE_DIR, TRANSFORM)
    print(f"Images found: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    embeddings, filenames = [], []

    model.eval()
    with torch.no_grad():
        for imgs, stems in tqdm(loader, desc="Encoding"):
            emb = model(imgs.to(device))
            embeddings.append(emb.cpu().numpy())
            filenames.extend(stems)

    emb_arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    fn_arr  = np.array(filenames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "messidor_retfound.npy",  emb_arr)
    np.save(OUT_DIR / "messidor_filenames.npy", fn_arr)

    print(f"\nSaved embeddings: {emb_arr.shape}")
    print(f"Saved filenames:  {fn_arr.shape} — sample: {fn_arr[:3]}")


if __name__ == "__main__":
    main()
