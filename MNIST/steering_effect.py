import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import visualizations  # noqa: F401 -- side effect only: applies the shared matplotlib style
from distributions import DIM, H1_VARIANTS, reward, IMG_SIZE
from diffusion import DiffusionModel, sample_FK

SEED             = 1337
N_SAMPLE         = 50 * 2
N_STEPS          = 100
N_PROBES         = 10
XI               = 4
RESAMPLING_TIMES = {}
STEERING_TIMES   = {round(0.02 * k, 2) for k in range(51)}
H1_VARIANT       = "sum"   # "ratio" or "sum"
log_h1_fn, dlog_p_dlog_h1_fn = H1_VARIANTS[H1_VARIANT]

KAPPAS = [100, 0.003, 0.002, 0.001, 0.0001, 0.000001]

MODELS_DIR = 'models'
DEVICE = 'cpu'
os.makedirs('Figures', exist_ok=True)

model1 = DiffusionModel(device=DEVICE, dim=DIM)
model1.load(os.path.join(MODELS_DIR, 'agent1.pt'))
model2 = DiffusionModel(device=DEVICE, dim=DIM)
model2.load(os.path.join(MODELS_DIR, 'agent2.pt'))
models = [model1, model2]

fig, axes = plt.subplots(1, len(KAPPAS), figsize=(4 * len(KAPPAS), 4))

for ax, kappa in zip(axes, KAPPAS):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Sampling via Feynman-Kac ({H1_VARIANT}, steered, kappa={kappa}) ...")
    samples, log_w, agent_ids, log_p_list = sample_FK(
        models            = models,
        n                 = N_SAMPLE,
        n_steps           = N_STEPS,
        n_probes          = N_PROBES,
        resampling_times  = RESAMPLING_TIMES,
        steering_times    = STEERING_TIMES,
        log_h1            = log_h1_fn,
        dlog_p_dlog_h1    = dlog_p_dlog_h1_fn,
        reward_fn         = reward,
        kappa             = kappa,
        xi                = XI,
    )
    best_idx   = int(np.argmax(log_w))
    best_image = ((samples[best_idx].reshape(IMG_SIZE, IMG_SIZE) + 1.0) / 2.0).clip(0, 1)

    ax.imshow(best_image, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(rf'$\kappa = {kappa:g}$', fontsize=26)

fig.tight_layout()
fig.savefig('Figures/mnist_steering_effect.pdf', bbox_inches='tight')
plt.show()
