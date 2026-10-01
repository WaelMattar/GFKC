import os
import time
import numpy as np
import torch

from distributions import DIGITS_1, DIGITS_2, DIM, H1_VARIANTS, reward, MEAN_9, IMG_SIZE
from diffusion import DiffusionModel, sample_FK
from visualizations import (plot_agent_grid, plot_all_particles, plot_log_w_hist, plot_reward_hist,
                             plot_log_w_vs_reward, plot_best_vs_reference)

# Seed for randomness
SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.makedirs('Figures', exist_ok=True)

# Diffusion parameters
N_PER_AGENT      = 100
N_SAMPLE         = 50 * 2
N_STEPS          = 100
N_PROBES         = 10
XI               = 4
RESAMPLING_TIMES = {}

# Target distribution parameters
H1_VARIANT       = "ratio"   # "ratio" or "sum"
log_h1_fn, dlog_p_dlog_h1_fn = H1_VARIANTS[H1_VARIANT]

# Steering parameters
STEERING         = True
KAPPA            = 0.000001
STEERING_TIMES   = {round(0.02 * k, 2) for k in range(51)} if STEERING else None

MODELS_DIR = 'models'
DEVICE = 'cpu'
print(f"Device: {DEVICE}")
t_start = time.time()

# Load agents
model1 = DiffusionModel(device=DEVICE, dim=DIM)
model1.load(os.path.join(MODELS_DIR, 'agent1.pt'))
print(f"Loaded agent 1 (digits {DIGITS_1})")
model2 = DiffusionModel(device=DEVICE, dim=DIM)
model2.load(os.path.join(MODELS_DIR, 'agent2.pt'))
print(f"Loaded agent 2 (digits {DIGITS_2})")
models = [model1, model2]
# Testing vanilla agent sampling
print(f"\nSampling {N_PER_AGENT} images per agent...")
samples1 = model1.sample(N_PER_AGENT, xi=XI)
samples2 = model2.sample(N_PER_AGENT, xi=XI)

print(f"\nSampling via Feynman-Kac ({H1_VARIANT}{', steered' if STEERING else ''}) ...")
samples, log_w, agent_ids, log_p_list = sample_FK(
    models = models,
    n = N_SAMPLE,
    n_steps = N_STEPS,
    n_probes = N_PROBES,
    resampling_times = RESAMPLING_TIMES,
    steering_times = STEERING_TIMES,
    log_h1 = log_h1_fn,
    dlog_p_dlog_h1 = dlog_p_dlog_h1_fn,
    reward_fn = reward if STEERING else None,
    kappa = KAPPA,
    xi = XI,
)

# Calculate elapsed time
elapsed = time.time() - t_start
hours, rem = divmod(int(elapsed), 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTotal execution time: {hours:02d}h {minutes:02d}m {seconds:02d}s")

rewards = reward(torch.from_numpy(samples)).numpy()

# Save and show results
plot_agent_grid(samples1, f'Agent 1 — digits {DIGITS_1}', save_path='Figures/agent_1_samples.pdf', img_size=IMG_SIZE, nrows=10, ncols=10)
plot_agent_grid(samples2, f'Agent 2 — digits {DIGITS_2}', save_path='Figures/agent_2_samples.pdf', img_size=IMG_SIZE, nrows=10, ncols=10)
plot_log_w_hist(log_weights=log_w, agent_ids=agent_ids)
plot_all_particles(samples, log_w, agent_ids, rewards, img_size=IMG_SIZE)

if STEERING:
    plot_reward_hist(rewards=rewards, agent_ids=agent_ids)
    plot_log_w_vs_reward(log_weights=log_w, rewards=rewards, agent_ids=agent_ids)
    best_idx = int(np.argmax(log_w))
    best_flipped = samples[best_idx].reshape(IMG_SIZE, IMG_SIZE)[::-1, ::-1].reshape(-1)
    plot_best_vs_reference(best_flipped, MEAN_9.numpy(), img_size=IMG_SIZE)
