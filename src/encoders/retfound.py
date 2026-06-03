"""
RETFound encoder wrapper.
Loads the pretrained RETFound colour fundus model from HuggingFace and
exposes encode() → (N, D) embeddings.

Model: openmedlab/RETFound_cfp (colour fundus photography, ViT-Large)
Paper: RETFound — a foundation model for retinal imaging (Nature 2023)

Use cases:
  EyePACS / Messidor  → classification and representation baselines
  OLIVES fundus       → temporal sequences for dynamics modeling
  OLIVES OCT          → may need separate OCT-specific encoder (TBD)

Embedding dim: check at runtime with probe_embedding_dim() — likely 1024
for ViT-Large but not hardcoded.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


RETFOUND_MEAN = [0.485, 0.456, 0.406]
RETFOUND_STD  = [0.229, 0.224, 0.225]

RETFOUND_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=RETFOUND_MEAN, std=RETFOUND_STD),
])


def load_retfound(device: str = "cpu") -> nn.Module:
    """Download and return RETFound model (frozen).

    Uses bitfount/RETFound_MAE — public mirror of the official model,
    loadable directly via timm without HuggingFace authentication.
    Official gated source: YukunZhou/RETFound_mae_natureCFP (requires HF login).
    """
    try:
        import timm
    except ImportError:
        raise ImportError("pip install timm")

    model = timm.create_model(
        "hf_hub:bitfount/RETFound_MAE",
        pretrained=True,
        num_classes=0,  # remove classification head → returns embeddings
    )

    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def probe_embedding_dim(model: nn.Module, device: str = "cpu") -> int:
    """Run one dummy image through the model and return embedding dimension."""
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        emb = model(dummy)
    dim = emb.shape[-1]
    print(f"RETFound embedding dim: {dim}")
    return dim


@torch.no_grad()
def encode_dataset(
    dataset,
    model: nn.Module,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> np.ndarray:
    """Encode all images in a Dataset, return (N, D) numpy array."""
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    embeddings = []
    for batch in tqdm(loader, desc="Encoding"):
        imgs = batch[0].to(device)
        emb = model(imgs)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def cache_embeddings(
    dataset,
    model: nn.Module,
    out_path: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode dataset and cache to disk. Loads from cache if already exists."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Loading cached embeddings from {out}")
        return np.load(out)
    print(f"Encoding {len(dataset)} images...")
    embs = encode_dataset(dataset, model, device=device, batch_size=batch_size)
    np.save(out, embs)
    print(f"Saved: {out} — shape {embs.shape}")
    return embs
