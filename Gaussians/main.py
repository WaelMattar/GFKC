import os
import time
import numpy as np
import torch

from diffusion import GaussianModel, sample_individual, sample_FK
from distributions import MU_1, MU_2, gaussian_pdf, H1_VARIANTS, H1_RAW, H1_LABELS, reward
from visualizations import (normalize, kde_on_grid, grid_pts, gx, gy, GRID_RES,
                             plot_comparison, plot_individual_models, plot_agent_evolution,
                             plot_unweighted_kde_agents, plot_log_weights_per_agent)

# Seed for randomness
SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs('Figures', exist_ok=True)

# Diffusion parameters
N_SAMPLE         = 100000 * 2
N_STEPS          = 100
XI               = 0
SNAPSHOT_TIMES   = (0.0, 0.5, 0.7, 0.85, 1.0)
RESAMPLING_TIMES = {}

# Target distribution parameters
H1_VARIANT       = "ratio"   # "ratio" or "symmetric"
log_h1_fn, dlog_p_dlog_h1_fn = H1_VARIANTS[H1_VARIANT]

# Steering parameters
STEERING         = False
KAPPA            = 1
STEERING_TIMES   = {round(0.02 * k, 2) for k in range(51)} if STEERING else None

t_start = time.time()

# Analytical models: no training needed — the score is known in closed form
print("\n[1/2] Instantiating analytical model for p_1 = N(mu_1, I) ...")
model1 = GaussianModel(MU_1)

print("\n[2/2] Instantiating analytical model for p_2 = N(mu_2, I) ...")
model2 = GaussianModel(MU_2)

models = [model1, model2]

print("\nSampling individual models ...")
s1 = sample_individual(model1, n=N_SAMPLE, n_steps=N_STEPS, xi=XI)
s2 = sample_individual(model2, n=N_SAMPLE, n_steps=N_STEPS, xi=XI)
ind1 = normalize(kde_on_grid(s1, np.ones(N_SAMPLE) / N_SAMPLE))
ind2 = normalize(kde_on_grid(s2, np.ones(N_SAMPLE) / N_SAMPLE))

print(f"\nSampling via FK ({H1_VARIANT}{', steered' if STEERING else ''}) target ...")
samples, log_w, agent_ids, agent_snapshots = sample_FK(
    models,
    n                = N_SAMPLE,
    n_steps          = N_STEPS,
    snapshot_times   = SNAPSHOT_TIMES,
    resampling_times = RESAMPLING_TIMES,
    steering_times   = STEERING_TIMES,
    log_h1           = log_h1_fn,
    dlog_p_dlog_h1   = dlog_p_dlog_h1_fn,
    reward_fn        = reward if STEERING else None,
    kappa            = KAPPA,
    xi               = XI,
)
log_w -= log_w.max()
w      = np.exp(log_w); w /= w.sum()
print(f"  ESS = {1.0 / (w**2).sum():.1f} / {N_SAMPLE}  (pooled over {len(np.unique(agent_ids))} agents)")
combined = normalize(kde_on_grid(samples, w))

snapshot_densities = []
for t_val, per_agent in agent_snapshots:
    densities_k = []
    for x_i, lw_i in per_agent:
        lw_i = lw_i - lw_i.max()
        w_i  = np.exp(lw_i); w_i /= w_i.sum()
        densities_k.append(normalize(kde_on_grid(x_i, w_i)))
    snapshot_densities.append((t_val, densities_k))

# True target evaluated analytically on the grid (matches whichever H1_VARIANT was selected)
p1_grid  = gaussian_pdf(grid_pts.T, MU_1).reshape(GRID_RES, GRID_RES)
p2_grid  = gaussian_pdf(grid_pts.T, MU_2).reshape(GRID_RES, GRID_RES)
h1_raw   = H1_RAW[H1_VARIANT]
true_pdf    = normalize(h1_raw(p1_grid, p2_grid))
naive_ratio = normalize(h1_raw(ind1, ind2))

eps      = 1e-300
kl       = np.trapezoid(np.trapezoid(naive_ratio * np.log(np.maximum(naive_ratio, eps) / np.maximum(combined, eps)), gx, axis=1), gy)
kl_true  = np.trapezoid(np.trapezoid(true_pdf * np.log(np.maximum(true_pdf, eps) / np.maximum(combined, eps)), gx, axis=1), gy)
print(f"\nKL(naive_ratio || q) = {kl:.4f}")
print(f"KL(true_pdf    || q) = {kl_true:.4f}")

elapsed = time.time() - t_start
hours, rem = divmod(int(elapsed), 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTotal execution time: {hours:02d}h {minutes:02d}m {seconds:02d}s")

K = len(models)
log_w_list_final = [log_w[agent_ids == i] for i in range(K)]

plot_individual_models(ind1, ind2)
plot_comparison(combined, true_pdf, h1_label=H1_LABELS[H1_VARIANT])
plot_agent_evolution(snapshot_densities)
plot_unweighted_kde_agents(samples, agent_ids, log_w)
plot_log_weights_per_agent(log_w_list_final, label='Final')
