"""
True Latent ODE dynamics model — Tier 4 (src/dynamics/).

Architecture: Latent ODE (Chen et al. 2018, NeurIPS) adapted for next-visit CST prediction.

Unlike the ODE-RNN (latent_ode.py, Rubanova 2019) which is discriminative (point prediction),
this model is generative: it learns a distribution over the initial hidden state z0 and
outputs calibrated predictive uncertainty.

Forward pass:
  1. Backward RNN encoder: processes visit embeddings in reverse → context vector c
  2. Variational bottleneck: c → q(z0) = N(mu_z0, sigma_z0²)
  3. Sample z0 via reparameterization (or use mu_z0 at eval time)
  4. ODE integration: odeint(f, z0, t_grid) → z(t) at each prediction time
  5. Decode: z(t) → mu_pred, sigma_pred (predictive distribution over next-visit CST)

Training objective: ELBO = reconstruction NLL + beta * KL(q(z0) || N(0, I))
  - KL beta warmup from 0 → 1 over first N epochs (avoids posterior collapse)
  - Collapse criterion: KL < 0.01 nats after warmup → stop, report failure (Decision #23)

Pre-committed evaluation bars (Decision #23, human confirmed):
  - 90% CI coverage >= 0.80
  - KL > 0.10 nats
  - RMSE <= 85 um

Input shape:  (batch, seq_len, emb_dim) — RETFound embeddings, zero-padded
Time input:   (batch, seq_len)          — per-step delta_t; None = ordinal 1.0
Outputs:      mu_pred    (batch, seq_len, 1)
              sigma_pred (batch, seq_len, 1)
              mu_z0      (batch, latent_dim)
              sigma_z0   (batch, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    dz/dt = f(z): autonomous MLP.

    Tanh activations keep the vector field bounded, which improves numerical
    stability with the dopri5 adaptive solver.
    No time dependence: disease dynamics depend on state, not absolute visit index.
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
        return self.net(z)


class TrueLatentODE(nn.Module):
    """
    Generative Latent ODE for irregular-time CST trajectory prediction.

    Args:
        emb_dim:        Input embedding dimension (1024 for RETFound).
        latent_dim:     Dimension of latent state z.
        encoder_hidden: Hidden units in the backward GRU encoder.
        ode_hidden:     Hidden units per layer in the ODE function MLP.
        dropout:        Dropout applied to z before decoding.
        rtol, atol:     ODE solver tolerances (tighten to 1e-5/1e-6 for final eval).
    """

    def __init__(
        self,
        emb_dim:        int   = 1024,
        latent_dim:     int   = 32,
        encoder_hidden: int   = 64,
        ode_hidden:     int   = 64,
        dropout:        float = 0.2,
        rtol:           float = 1e-3,
        atol:           float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.rtol = rtol
        self.atol = atol

        # Backward encoder: projects embeddings → GRU context
        self.input_proj  = nn.Linear(emb_dim, encoder_hidden)
        self.encoder_rnn = nn.GRU(encoder_hidden, encoder_hidden, batch_first=True)

        # Variational bottleneck: context → q(z0)
        self.mu_net    = nn.Linear(encoder_hidden, latent_dim)
        self.sigma_net = nn.Linear(encoder_hidden, latent_dim)

        # ODE and decoder
        self.ode_func      = ODEFunc(latent_dim, ode_hidden)
        self.dropout       = nn.Dropout(dropout)
        self.decoder_mu    = nn.Linear(latent_dim, 1)
        self.decoder_sigma = nn.Linear(latent_dim, 1)

    def _reverse_sequences(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse only the valid portion of each sequence, leaving padding unchanged.

        Args:
            x:       [batch, seq_len, dim]
            lengths: [batch]

        Returns:
            [batch, seq_len, dim]
        """
        out = x.clone()
        for i, l in enumerate(lengths):
            out[i, : l.item()] = x[i, : l.item()].flip(0)
        return out

    def encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward RNN encode: infer q(z0) from the full observation sequence.

        Runs a GRU over the time-reversed sequence so the final hidden state
        is most informed by the earliest observation — the right prior for z0 at t=0.

        Args:
            x:       [batch, seq_len, emb_dim]
            lengths: [batch]

        Returns:
            mu_z0:    [batch, latent_dim]
            sigma_z0: [batch, latent_dim]  — always positive (Softplus + eps)
        """
        h = self.input_proj(x)                       # (batch, seq_len, encoder_hidden)
        h_rev = self._reverse_sequences(h, lengths)  # flip valid portion in time

        packed = pack_padded_sequence(
            h_rev, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder_rnn(packed)            # h_n: (1, batch, encoder_hidden)
        c = h_n.squeeze(0)                           # (batch, encoder_hidden)

        mu_z0    = self.mu_net(c)                            # (batch, latent_dim)
        sigma_z0 = F.softplus(self.sigma_net(c)) + 1e-5     # (batch, latent_dim)
        return mu_z0, sigma_z0

    def forward(
        self,
        x:           torch.Tensor,
        lengths:     torch.Tensor,
        delta_t_seq: torch.Tensor | None = None,
        sample:      bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x:            [batch, seq_len, emb_dim] — embeddings, zero-padded
            lengths:      [batch] — valid sequence length per eye
            delta_t_seq:  [batch, seq_len] — per-step time gap (None = ordinal 1.0)
            sample:       If True, sample z0 via reparameterization (training).
                          If False, use mu_z0 directly (evaluation — deterministic RMSE).

        Returns:
            mu_pred:    [batch, seq_len, 1] — predicted mean CST (normalised)
            sigma_pred: [batch, seq_len, 1] — predicted std (aleatoric uncertainty)
            mu_z0:      [batch, latent_dim] — posterior mean of initial state
            sigma_z0:   [batch, latent_dim] — posterior std of initial state
        """
        batch, seq_len, _ = x.shape
        device = x.device

        mu_z0, sigma_z0 = self.encode(x, lengths)

        if sample:
            eps = torch.randn_like(mu_z0)
            z0 = mu_z0 + sigma_z0 * eps
        else:
            z0 = mu_z0                                   # deterministic eval path

        # Integrate ODE per sequence (variable time grids require individual calls)
        all_z = torch.zeros(batch, seq_len, self.latent_dim, device=device)

        for i in range(batch):
            L = int(lengths[i].item())
            if delta_t_seq is not None:
                dts = delta_t_seq[i, :L].float().clamp(min=1e-3)
                t_grid = torch.cat([
                    torch.zeros(1, device=device, dtype=torch.float32),
                    dts.cumsum(0),
                ])                                       # (L+1,) strictly increasing
            else:
                t_grid = torch.arange(
                    L + 1, dtype=torch.float32, device=device
                )

            z_traj = odeint(
                self.ode_func,
                z0[i : i + 1],                          # (1, latent_dim)
                t_grid,
                method="rk4",
                options={"step_size": 0.5},
            )                                            # (L+1, 1, latent_dim)

            # z_traj[0] = z0 at t=0; z_traj[1:] = z at each prediction time
            all_z[i, :L] = z_traj[1:].squeeze(1)

        z_drop     = self.dropout(all_z)
        mu_pred    = self.decoder_mu(z_drop)                      # (batch, seq_len, 1)
        sigma_pred = F.softplus(self.decoder_sigma(z_drop)) + 1e-5  # (batch, seq_len, 1)

        return mu_pred, sigma_pred, mu_z0, sigma_z0

    @staticmethod
    def elbo_loss(
        mu_pred:    torch.Tensor,
        sigma_pred: torch.Tensor,
        targets:    torch.Tensor,
        lengths:    torch.Tensor,
        mu_z0:      torch.Tensor,
        sigma_z0:   torch.Tensor,
        beta:       float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO = Gaussian reconstruction NLL + beta * KL(q(z0) || N(0, I)).

        Args:
            mu_pred:    [batch, seq_len, 1]
            sigma_pred: [batch, seq_len, 1]
            targets:    [batch, seq_len] — normalised CST targets
            lengths:    [batch]
            mu_z0:      [batch, latent_dim]
            sigma_z0:   [batch, latent_dim]
            beta:       KL weight (0 at warmup start, 1 after warmup)

        Returns:
            (total_loss, recon_loss, kl_loss) — all scalar tensors
        """
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, : l.item()] = True

        # Gaussian NLL on valid steps only
        dist = Normal(mu_pred.squeeze(-1), sigma_pred.squeeze(-1))
        recon_loss = -dist.log_prob(targets)[mask].mean()

        # KL(N(mu_z0, sigma_z0²) || N(0, I)) = 0.5 * sum(sigma² + mu² - 1 - 2*log(sigma))
        kl_loss = 0.5 * (
            sigma_z0 ** 2 + mu_z0 ** 2 - 1.0 - 2.0 * sigma_z0.log()
        ).sum(-1).mean()

        return recon_loss + beta * kl_loss, recon_loss, kl_loss
