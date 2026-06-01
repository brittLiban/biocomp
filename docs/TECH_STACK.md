> Parent: CLAUDE.md · Related: docs/ARCHITECTURE.md, docs/CODE_STYLE.md
> This doc: the build stack. Read before writing any file.
> Changes: only when the stack genuinely changes (new library, version bump).

# Tech Stack

## Principle
Boring, well-supported, research-credible choices. Optimize for reproducibility and
AI-assisted development speed. When in doubt, pick the standard tool.

## Core
| Layer | Choice | Version | Why |
|---|---|---|---|
| Language | Python | 3.11+ | ML standard; massive ecosystem |
| ML framework | PyTorch | 2.x | Research standard; better than TF for this work |
| ODE solver | torchdiffeq | latest | Battle-tested Neural/Latent ODEs |
| Probabilistic | Pyro / NumPyro | latest | Bayesian modeling; future HAWVA |
| Image | Pillow, OpenCV | latest | Standard image handling |
| Data | pandas, polars, numpy | latest | Standard data tooling |
| Encoder | RETFound (via HuggingFace) | — | Pretrained retinal foundation model |
| Experiment tracking | Weights & Biases | latest | Free tier; all runs logged here |
| Version control | Git + GitHub | — | Public repo for credibility |
| Compute | Google Colab → GCP/AWS | — | Free tier first, scale with grants |
| Dev environment | VS Code + Claude Code | — | AI-accelerated development |

## Year 2+ (do NOT add yet)
FastAPI (API), PostgreSQL (production data), Docker (deployment). Not needed for the
prototype. Adding them now is premature.

## What NOT To Use
- TensorFlow/Keras — we are a PyTorch shop. Do not mix.
- localStorage/sessionStorage in any artifact — unsupported.
- Heavy MLOps platforms — overkill at prototype scale.
- Any library not listed here without logging a decision in DECISIONS.md first.

## Environment
- All dependencies pinned in `requirements.txt` (or `pyproject.toml`).
- Random seeds set for every experiment (numpy, torch, python).
- Environment specs committed so any run is reproducible.
