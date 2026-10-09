import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import distributions as dist
import visualizations  # noqa: F401 -- side effect only: applies the shared matplotlib style
from distributions import DIM, H1_VARIANTS
from diffusion import DiffusionModel, sample_FK

SEED             = 1337
N_SAMPLE         = 50 * 2
N_STEPS          = 100
N_PROBES         = 10
XI               = 4
RESAMPLING_TIMES = {}

POW_VALUES = [1, 2, 3, 10]

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

# --- 2x2 grid (pow=1,2 top row; pow=3,4 bottom row), each panel following the same
#     aesthetic as plot_log_w_hist in visualizations.py (cyan/magenta, no edges, forced
#     rescale + adaptive density for sparse agents). Only the top-left panel (pow=1)
#     carries a legend.
AGENT_COLORS = ['cyan', 'magenta']
FORCE_THRESHOLD = 10
FORCE_FRACTION = 0.1

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (pow_value, log_weights, agent_ids) in zip(axes.ravel(), results):
    groups = [log_weights[agent_ids == 0], log_weights[agent_ids == 1]]
    bin_edges = np.histogram_bin_edges(log_weights, bins=len(log_weights))
    bin_width = bin_edges[1] - bin_edges[0]

    counts_n_peak = []
    for data in groups:
        n = len(data)
        peak = np.histogram(data, bins=bin_edges)[0].max() / n if n > 0 else 0.0
        counts_n_peak.append((n, peak))

    big_peaks = [peak for n, peak in counts_n_peak if n >= FORCE_THRESHOLD]
    FIGURE_HEIGHT = max(big_peaks) if big_peaks else max(peak for _, peak in counts_n_peak)

    scales = [(FORCE_FRACTION * FIGURE_HEIGHT) / peak if n < FORCE_THRESHOLD and peak > 0 else 1.0
              for n, peak in counts_n_peak]
    weights = [np.full(n, scale / n) if n > 0 else np.array([])
               for (n, _), scale in zip(counts_n_peak, scales)]

    ax.hist(groups,
            bins=bin_edges,
            color=AGENT_COLORS,
            label=['First agent', 'Second agent'],
            weights=weights)

    STD_FALLBACK = 0.01 * (log_weights.max() - log_weights.min())
    xs = np.linspace(log_weights.min(), log_weights.max(), 500)
    for data, color, (n, _), scale in zip(groups, AGENT_COLORS, counts_n_peak, scales):
        if n == 0:
            continue
        if n >= FORCE_THRESHOLD and np.std(data) > 0:
            density = gaussian_kde(data, bw_method=0.1)(xs)
        else:
            density = np.zeros_like(xs)
            for mu in data:
                density += np.exp(-0.5 * ((xs - mu) / STD_FALLBACK) ** 2) / (STD_FALLBACK * np.sqrt(2 * np.pi))
            density /= n
        ax.plot(xs, density * bin_width * scale, color=color, linewidth=3)

    ax.set_xlabel(f'pow = {pow_value:g}', fontsize=20)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=20)
    if pow_value == POW_VALUES[0]:
        ax.legend(fontsize=16)

fig.tight_layout()
fig.savefig('Figures/sharpening_sum_effect.pdf', bbox_inches='tight')
