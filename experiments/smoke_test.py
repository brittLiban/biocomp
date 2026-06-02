"""
Smoke test — verifies reproducible infrastructure end-to-end.
Run twice with the same seed and confirm identical loss values.
Usage: python experiments/smoke_test.py --seed 42
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.seeds import set_seed

BATCH_SIZE = 8
INPUT_DIM = 64
HIDDEN_DIM = 32
OUTPUT_DIM = 4
STEPS = 10


def build_model(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def main(seed: int):
    set_seed(seed)

    run = wandb.init(
        project="synapse-v1",
        name=f"smoke-test-seed{seed}",
        config={
            "seed": seed,
            "batch_size": BATCH_SIZE,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "steps": STEPS,
        },
        tags=["smoke-test"],
    )

    model = build_model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses = []
    for step in range(STEPS):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        y = torch.randn(BATCH_SIZE, OUTPUT_DIM)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        wandb.log({"loss": loss.item(), "step": step})
        print(f"step {step:02d} | loss: {loss.item():.6f}")

    final_loss = losses[-1]
    wandb.summary["final_loss"] = final_loss
    wandb.finish()

    print(f"\nFinal loss (seed={seed}): {final_loss:.6f}")
    print(f"W&B run: {run.url}")
    return final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
