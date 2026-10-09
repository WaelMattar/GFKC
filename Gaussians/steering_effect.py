import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion import GaussianModel, sample_FK
from distributions import MU_1, MU_2, H1_VARIANTS, reward
from visualizations import normalize, kde_on_grid, Gx, Gy, _style_ax

SEED             = 1337
N_SAMPLE         = 100000 * 2
N_STEPS          = 100
XI               = 0
RESAMPLING_TIMES = {}
STEERING_TIMES   = {round(0.02 * k, 2) for k in range(51)}
H1_VARIANT       = "symmetric"   # "ratio" or "symmetric"
log_h1_fn, dlog_p_dlog_h1_fn = H1_VARIANTS[H1_VARIANT]

KAPPAS = [1_000_000, 10, 3, 1, 0.5]

os.makedirs('Figures', exist_ok=True)

fig, axes = plt.subplots(1, len(KAPPAS), figsize=(4 * len(KAPPAS), 4))

for ax, kappa in zip(axes, KAPPAS):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    models = [GaussianModel(MU_1), GaussianModel(MU_2)]
    samples, log_w, agent_ids, _ = sample_FK(
        models,
        n                = N_SAMPLE,
        n_steps          = N_STEPS,
        snapshot_times   = None,
        resampling_times = RESAMPLING_TIMES,
        steering_times   = STEERING_TIMES,
        log_h1           = log_h1_fn,
        dlog_p_dlog_h1   = dlog_p_dlog_h1_fn,
        reward_fn        = reward,
        kappa            = kappa,
        xi               = XI,
    )
    log_w = log_w - log_w.max()
    w     = np.exp(log_w); w /= w.sum()
    density = normalize(kde_on_grid(samples, w))

    ax.contourf(Gx, Gy, density, levels=30, cmap='viridis')
    _style_ax(ax)
    ax.set_ylabel('')
    ax.set_xlabel(rf'$\kappa = {kappa:g}$', fontsize=26)
    ax.tick_params(axis='both', labelsize=16)
    print(f"  kappa = {kappa:<10g} ESS = {1.0 / (w**2).sum():.1f} / {N_SAMPLE}")

fig.tight_layout()
fig.savefig('Figures/gaussian_steering_effect.pdf', bbox_inches='tight')
plt.show()
