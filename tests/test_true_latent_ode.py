"""
Sanity checks for TrueLatentODE — shape, loss, and gradient flow.
Does not require OLIVES data or W&B.

Run: python -m pytest tests/test_true_latent_ode.py -v
"""

import sys
sys.path.insert(0, "src")

import torch
import pytest
from dynamics.true_latent_ode import TrueLatentODE


BATCH      = 4
SEQ_LEN    = 8
EMB_DIM    = 1024
LATENT_DIM = 32


@pytest.fixture
def model():
    return TrueLatentODE(
        emb_dim        = EMB_DIM,
        latent_dim     = LATENT_DIM,
        encoder_hidden = 16,
        ode_hidden     = 16,
        dropout        = 0.0,
        rtol           = 1e-2,
        atol           = 1e-3,
    ).eval()


@pytest.fixture
def batch_data():
    torch.manual_seed(0)
    lengths     = torch.tensor([8, 6, 5, 3])
    x           = torch.randn(BATCH, SEQ_LEN, EMB_DIM)
    delta_t_seq = torch.ones(BATCH, SEQ_LEN)
    targets     = torch.randn(BATCH, SEQ_LEN)
    return x, targets, lengths, delta_t_seq


def test_output_shapes(model, batch_data):
    x, _, lengths, delta_t_seq = batch_data
    mu_pred, sigma_pred, mu_z0, sigma_z0 = model(
        x, lengths, delta_t_seq=delta_t_seq, sample=False
    )
    assert mu_pred.shape    == (BATCH, SEQ_LEN, 1)
    assert sigma_pred.shape == (BATCH, SEQ_LEN, 1)
    assert mu_z0.shape      == (BATCH, LATENT_DIM)
    assert sigma_z0.shape   == (BATCH, LATENT_DIM)


def test_sigma_positive(model, batch_data):
    x, _, lengths, delta_t_seq = batch_data
    _, sigma_pred, _, sigma_z0 = model(
        x, lengths, delta_t_seq=delta_t_seq, sample=False
    )
    assert (sigma_pred > 0).all(), "sigma_pred must be strictly positive"
    assert (sigma_z0   > 0).all(), "sigma_z0 must be strictly positive"


def test_elbo_loss_scalar(model, batch_data):
    x, targets, lengths, delta_t_seq = batch_data
    mu_pred, sigma_pred, mu_z0, sigma_z0 = model(
        x, lengths, delta_t_seq=delta_t_seq, sample=True
    )
    loss, recon, kl = TrueLatentODE.elbo_loss(
        mu_pred, sigma_pred, targets, lengths, mu_z0, sigma_z0, beta=1.0
    )
    assert loss.shape  == torch.Size([])
    assert recon.shape == torch.Size([])
    assert kl.shape    == torch.Size([])
    assert kl.item() >= 0.0, "KL must be non-negative"


def test_kl_warmup_zero(model, batch_data):
    x, targets, lengths, delta_t_seq = batch_data
    mu_pred, sigma_pred, mu_z0, sigma_z0 = model(
        x, lengths, delta_t_seq=delta_t_seq, sample=True
    )
    loss_beta0, recon, kl = TrueLatentODE.elbo_loss(
        mu_pred, sigma_pred, targets, lengths, mu_z0, sigma_z0, beta=0.0
    )
    # With beta=0, total loss should equal recon loss
    assert torch.isclose(loss_beta0, recon, atol=1e-5)


def test_gradients_flow(batch_data):
    torch.manual_seed(1)
    x, targets, lengths, delta_t_seq = batch_data

    m = TrueLatentODE(
        emb_dim=EMB_DIM, latent_dim=LATENT_DIM,
        encoder_hidden=16, ode_hidden=16, dropout=0.0,
        rtol=1e-2, atol=1e-3,
    ).train()

    mu_pred, sigma_pred, mu_z0, sigma_z0 = m(
        x, lengths, delta_t_seq=delta_t_seq, sample=True
    )
    loss, _, _ = TrueLatentODE.elbo_loss(
        mu_pred, sigma_pred, targets, lengths, mu_z0, sigma_z0, beta=1.0
    )
    loss.backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


def test_eval_deterministic(model, batch_data):
    x, _, lengths, delta_t_seq = batch_data
    out1 = model(x, lengths, delta_t_seq=delta_t_seq, sample=False)[0]
    out2 = model(x, lengths, delta_t_seq=delta_t_seq, sample=False)[0]
    assert torch.allclose(out1, out2), "eval (sample=False) must be deterministic"


def test_ordinal_fallback(model, batch_data):
    x, _, lengths, _ = batch_data
    # Should run without error when delta_t_seq=None (ordinal spacing)
    mu_pred, sigma_pred, mu_z0, sigma_z0 = model(
        x, lengths, delta_t_seq=None, sample=False
    )
    assert mu_pred.shape == (BATCH, SEQ_LEN, 1)
