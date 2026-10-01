import os
import time
import torch
import numpy as np

from distributions import load_agent_data, DIGITS_1, DIGITS_2, DIM
from diffusion import DiffusionModel

SEED = 42
N_ITERS = 300000
BATCH_SIZE = 512
LR = 2e-4
HIDDEN = 2048
MODELS_DIR = 'models'

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(f"Device: {DEVICE}")

os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading MNIST (train + test, enriched) ...")
# Each digit loaded separately so training can draw exactly-equal-probability
# batches per digit, regardless of the raw class-count imbalance.
data1a = load_agent_data([DIGITS_1[0]], include_test=True)
data1b = load_agent_data([DIGITS_1[1]], include_test=True)
data2a = load_agent_data([DIGITS_2[0]], include_test=True)
data2b = load_agent_data([DIGITS_2[1]], include_test=True)
print(f"  Agent 1 digits {DIGITS_1}: {len(data1a):,} + {len(data1b):,} images, dim={data1a.shape[1]}")
print(f"  Agent 2 digits {DIGITS_2}: {len(data2a):,} + {len(data2b):,} images, dim={data2a.shape[1]}")

def _fmt(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

t0 = time.time()

model1 = DiffusionModel(device=DEVICE, dim=DIM, hidden=HIDDEN)
print(f"\nTraining agent 1 (digits {DIGITS_1}, equal probability) for {N_ITERS} iterations ...")
t1 = time.time()
model1.train_balanced(data1a, data1b, n_iters=N_ITERS, batch_size=BATCH_SIZE, lr=LR)
print(f"  Agent 1 training time: {_fmt(time.time() - t1)}")
torch.save(model1.net.state_dict(), os.path.join(MODELS_DIR, 'agent1.pt'))
print(f"  Saved -> {MODELS_DIR}/agent1.pt")

model2 = DiffusionModel(device=DEVICE, dim=DIM, hidden=HIDDEN)
print(f"\nTraining agent 2 (digits {DIGITS_2}, equal probability) for {N_ITERS} iterations ...")
t2 = time.time()
model2.train_balanced(data2a, data2b, n_iters=N_ITERS, batch_size=BATCH_SIZE, lr=LR)
print(f"  Agent 2 training time: {_fmt(time.time() - t2)}")
torch.save(model2.net.state_dict(), os.path.join(MODELS_DIR, 'agent2.pt'))
print(f"  Saved -> {MODELS_DIR}/agent2.pt")

print(f"\nTotal training time: {_fmt(time.time() - t0)}")
