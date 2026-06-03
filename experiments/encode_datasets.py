"""
Encode EyePACS and Messidor images with frozen RETFound.
Caches embeddings to data/processed/embeddings/.
Runtime: ~4-6 hours on CPU for EyePACS (35K images).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import wandb
from src.utils.seeds import set_seed
from src.encoders.retfound import load_retfound, probe_embedding_dim, cache_embeddings
from src.data.eyepacs import EyePACSDataset
from src.data.messidor import MessidorDataset

DEVICE = "cpu"
BATCH_SIZE = 16
SEED = 42

EYEPACS_IMG_DIR  = "data/raw/eyepacs/train/images/train"
EYEPACS_CSV      = "data/raw/eyepacs/trainLabels.csv"
MESSIDOR_IMG_DIR = "data/raw/messidor/images/IMAGES"
MESSIDOR_CSV     = "data/raw/messidor/messidor-2.csv"
OUT_DIR          = "data/processed/embeddings"


def main():
    set_seed(SEED)

    print("Loading RETFound...")
    model = load_retfound(device=DEVICE)
    dim = probe_embedding_dim(model, device=DEVICE)

    run = wandb.init(
        project="synapse-v1",
        name="retfound-encoding",
        config={"device": DEVICE, "batch_size": BATCH_SIZE,
                "embedding_dim": dim, "seed": SEED},
        tags=["encoding", "retfound"],
    )

    # --- EyePACS ---
    print("\nEncoding EyePACS...")
    eyepacs = EyePACSDataset(EYEPACS_IMG_DIR, EYEPACS_CSV)
    print(eyepacs)
    embs_eyepacs = cache_embeddings(
        eyepacs, model,
        out_path=f"{OUT_DIR}/eyepacs_retfound.npy",
        device=DEVICE, batch_size=BATCH_SIZE,
    )
    wandb.log({"eyepacs/n_images": len(eyepacs),
               "eyepacs/embedding_shape": str(embs_eyepacs.shape),
               "eyepacs/emb_mean": float(embs_eyepacs.mean()),
               "eyepacs/emb_std": float(embs_eyepacs.std())})
    print(f"EyePACS embeddings: {embs_eyepacs.shape}")

    # --- Messidor ---
    # Use a simple folder dataset — Messidor CSV has non-standard format (left;right pairs).
    # Labels are not needed for encoding; load all images from disk directly.
    print("\nEncoding Messidor-2...")
    from torch.utils.data import Dataset as TorchDataset
    from PIL import Image as PILImage

    class FolderDataset(TorchDataset):
        def __init__(self, folder, transform=None):
            from torchvision import transforms as T
            self.paths = sorted(
                p for p in Path(folder).rglob("*")
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            )
            self.transform = transform or RETFOUND_TRANSFORM
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = PILImage.open(self.paths[i]).convert("RGB")
            return self.transform(img), 0, str(self.paths[i].name)

    from src.encoders.retfound import RETFOUND_TRANSFORM
    messidor = FolderDataset(MESSIDOR_IMG_DIR)
    print(f"FolderDataset(n={len(messidor)}, dir={MESSIDOR_IMG_DIR})")
    embs_messidor = cache_embeddings(
        messidor, model,
        out_path=f"{OUT_DIR}/messidor_retfound.npy",
        device=DEVICE, batch_size=BATCH_SIZE,
    )
    wandb.log({"messidor/n_images": len(messidor),
               "messidor/embedding_shape": str(embs_messidor.shape),
               "messidor/emb_mean": float(embs_messidor.mean()),
               "messidor/emb_std": float(embs_messidor.std())})
    print(f"Messidor embeddings: {embs_messidor.shape}")

    wandb.finish()
    print("\nAll done. Embeddings cached to", OUT_DIR)


if __name__ == "__main__":
    main()
