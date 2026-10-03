import os
import numpy as np
import torch

import distributions as dist
from distributions import DIM, H1_VARIANTS
from diffusion import DiffusionModel, sample_FK
from visualizations import plot_sharpening_sum_effect

SEED             = 1337
N_SAMPLE         = 50 * 2
N_STEPS          = 100
N_PROBES         = 10
XI               = 4
RESAMPLING_TIMES = {}

POW_VALUES = [1, 2, 3]

MODELS_DIR = 'models'
DEVICE = 'cpu'
os.makedirs('Figures', exist_ok=True)

log_h1_fn, dlog_p_dlog_h1_fn = H1_VARIANTS['sum']

model1 = DiffusionModel(device=DEVICE, dim=DIM)
model1.load(os.path.join(MODELS_DIR, 'agent1.pt'))
model2 = DiffusionModel(device=DEVICE, dim=DIM)
model2.load(os.path.join(MODELS_DIR, 'agent2.pt'))
models = [model1, model2]

results = []
for pow_value in POW_VALUES:
    dist.H1_SUM_POW = pow_value
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Sampling via Feynman-Kac (sum, pow={pow_value}) ...")
    samples, log_w, agent_ids, log_p_list = sample_FK(
        models            = models,
        n                 = N_SAMPLE,
        n_steps           = N_STEPS,
        n_probes          = N_PROBES,
        resampling_times  = RESAMPLING_TIMES,
        log_h1            = log_h1_fn,
        dlog_p_dlog_h1    = dlog_p_dlog_h1_fn,
        xi                = XI,
    )
    results.append((pow_value, log_w, agent_ids))

plot_sharpening_sum_effect(results)
