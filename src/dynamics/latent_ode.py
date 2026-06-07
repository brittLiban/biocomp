"""
Latent ODE dynamics model — Tier 4 (src/dynamics/).

Architecture: ODE-RNN (Rubanova et al. 2019, NeurIPS)

  At each observed visit t:
    1. Encode embedding:   z_obs = encoder(emb_t)           [emb_dim → latent_dim]
    2. Assimilate into state: h_t = GRUCell(z_obs, h_{t-1}) [latent_dim → latent_dim]
    3. Integrate ODE:    h_{t+1} = odeint(f, h_t, [0, δt])[-1]
    4. Decode:         ĉst_{t+1} = decoder(h_{t+1})        [latent_dim → 1]
    5. Carry h_{t+1} as latent state for the next visit.

Why ODE-RNN and not a pure latent ODE encoder?
  A pure encoder (no GRU) would condition each CST prediction solely on the
  current visit's embedding and the ODE forward pass — no accumulated history.
  GRU-D and T-LSTM both maintain hidden state across visits; dropping the GRU
  update here would give the ODE less information than the baselines it must beat.
  The GRU update assimilates each new observation; the ODE models continuous
  dynamics between observations. Both are necessary.

Relationship to Chen et al. 2018:
  The original Latent ODE (Chen et al.) uses an RNN encoder run *backwards*
  to infer a posterior z_0, then integrates forward. That VAE framing is not
  used here — we are in discriminative mode (predict next-visit CST), not
  generative mode (reconstruct the sequence). The ODE-RNN variant
  (Rubanova et al. 2019) is the direct analogue for predictive tasks.

Input shape: (batch, seq_len, emb_dim)   — embeddings, one per visit
Output shape: (batch, seq_len, 1)         — predicted CST at next visit, normalised
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    dz/dt = f(z): autonomous MLP.

    Tanh activations are standard for neural ODEs — they keep the vector
    field bounded, which improves numerical stability with dopri5.
    No explicit time dependence: disease dynamics don't depend on absolute
    visit number, only on the current latent state.
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # t is required by torchdiffeq's interface; unused (autonomous system)
        return self.net(z)


class LatentODE(nn.Module):
    """
    ODE-RNN latent dynamics model for irregular time-series prediction.

    Args:
        emb_dim:    Input embedding dimension (1024 for RETFound).
        latent_dim: Latent state dimension.
        ode_hidden: Hidden units in each ODEFunc layer.
        dropout:    Applied to latent state before decoding.
        rtol, atol: ODE solver tolerances. 1e-3 / 1e-4 is standard for
                    training; tighten to 1e-5 / 1e-6 for final evaluation.
    """

    def __init__(
        self,
        emb_dim:    int   = 1024,
        latent_dim: int   = 32,
        ode_hidden: int   = 64,
        dropout:    float = 0.2,
        rtol:       float = 1e-3,
        atol:       float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.rtol = rtol
        self.atol = atol

        self.encoder    = nn.Linear(emb_dim, latent_dim)
        self.gru_update = nn.GRUCell(latent_dim, latent_dim)
        self.ode_func   = ODEFunc(latent_dim, ode_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.decoder    = nn.Linear(latent_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        delta_t_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (batch, seq_len, emb_dim) — one embedding per visit step.
                          Sequences shorter than seq_len are zero-padded.
            lengths:      (batch,) — true sequence length per eye (padding mask).
            delta_t_seq:  (batch, seq_len) — per-step integration interval.
                          If None, uses ordinal δt=1.0 at every step.
                          Real week gaps (normalized): Prime eyes have values in
                          {1.0, 2.0, 3.0, ...}; TREX eyes are always 1.0.
                          Batch-mean over non-padded elements is used at each step
                          so a single odeint call handles the whole batch.

        Returns:
            (batch, seq_len, 1) — predicted normalised CST at each next visit.
            Positions beyond lengths[i] are garbage; mask with lengths before loss.
        """
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.latent_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            emb_t = x[:, t, :]                              # (batch, emb_dim)
            z_obs = self.encoder(emb_t)                     # (batch, latent_dim)
            h     = self.gru_update(z_obs, h)               # (batch, latent_dim)

            if delta_t_seq is not None:
                # Batch-mean delta_t over non-padded positions at this step
                valid = lengths > t
                dt = delta_t_seq[:, t][valid].float().mean().item() if valid.any() else 1.0
                dt = max(dt, 1e-6)
            else:
                dt = 1.0

            t_span = torch.tensor([0.0, dt], device=x.device, dtype=x.dtype)
            h = odeint(
                self.ode_func, h, t_span,
                method="dopri5",
                rtol=self.rtol, atol=self.atol,
            )[-1]                                            # (batch, latent_dim)

            outputs.append(self.decoder(self.dropout(h)))   # (batch, 1)

        return torch.stack(outputs, dim=1)                  # (batch, seq_len, 1)
