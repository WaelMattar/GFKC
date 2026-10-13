import math
import torch
import numpy as np
import ot
import torch.nn.functional as F
from torchvision import datasets

IMG_SIZE = 28
DIGITS_1 = [1, 6]
DIGITS_2 = [1, 7]
DIM = 784
C = 100
LOG_C = math.log(C)
H1_SUM_POW = 1

def load_agent_data(digits):
    ds = datasets.MNIST(root='./data', train=True, download=True)
    mask = torch.zeros(len(ds.targets), dtype=torch.bool)
    for d in digits:
        mask |= (ds.targets == d)
    return ds.data[mask].float().view(-1, DIM) / 127.5 - 1.0

def log_h0(log_p):
    # h0 = geometric mean sqrt(p_1 * p_2)  ->  log_h0 = mean(log_p)
    return log_p.mean(dim=0)

def dlog_p_dlog_h0(log_p):
    # ∂(log h_0)/∂(log p_1) = ∂(log h_0)/∂(log p_2) = 1/2
    return torch.full(size=log_p.size(), fill_value=1/2)

def log_h1_ratio(log_p):
    # h1 = p_1 / (1 + C*p_2)
    # TARGET distribution: favors agent 1 over agent 2.
    return log_p[0] - F.softplus(log_p[1] + LOG_C)

def dlog_p_dlog_h1_ratio(log_p):
    # ∂(log h_1)/∂(log p_1) = 1
    # ∂(log h_1)/∂(log p_2) = - C*p_2 / (1 + C*p_2)
    d0 = torch.ones_like(log_p[0:1])
    d1 = -torch.sigmoid(log_p[1:2] + LOG_C)
    return torch.cat([d0, d1], dim=0)

def log_h1_sum(log_p):
    # h1 = (p_1 + p_2)^pow  (pow = H1_SUM_POW, 1 <= pow <= 3)
    return H1_SUM_POW * torch.logaddexp(log_p[0], log_p[1])

def dlog_p_dlog_h1_sum(log_p):
    # ∂(log h_1)/∂(log p_1) = pow * p_1/(p_1+p_2) ; ∂(log h_1)/∂(log p_2) = pow * p_2/(p_1+p_2)
    d0 = H1_SUM_POW * torch.sigmoid(log_p[0:1] - log_p[1:2])
    d1 = H1_SUM_POW * torch.sigmoid(log_p[1:2] - log_p[0:1])
    return torch.cat([d0, d1], dim=0)

# h1 toggle registry: pick H1_VARIANT in main.py, pass the (log_h1, dlog_p_dlog_h1)
H1_VARIANTS = {
    "ratio": (log_h1_ratio, dlog_p_dlog_h1_ratio),
    "sum":   (log_h1_sum,   dlog_p_dlog_h1_sum),
}

# Feynman-Kac anneal along the diffusion time (t = 0 noise, t = 1 data):
def A_0(t_val): return 1.0 - t_val
def A_1(t_val): return t_val
def A_0_prime(t_val): return -1.0
def A_1_prime(t_val): return 1.0

# Steering: interpolate from log(h0) to log(h1 * e^{r(x)/KAPPA}) instead of log(h1).
def _load_mean_9():
    ds = datasets.MNIST(root='./data', train=True, download=True)
    imgs = ds.data[ds.targets == 9].float().view(-1, DIM) / 127.5 - 1.0
    return imgs.mean(dim=0)

MEAN_9 = _load_mean_9()

_g = np.arange(IMG_SIZE) / (IMG_SIZE - 1)
_grid = np.stack(np.meshgrid(_g, _g, indexing="ij"), -1).reshape(-1, 2)
_GROUND_COST = ot.dist(_grid, _grid, metric="euclidean")

def _to_measure(img):
    a = np.clip((img + 1.0) / 2.0, 0, None)
    s = a.sum()
    return a / s if s > 0 else np.full_like(a, 1.0 / a.size)

_MEAN_9_MEASURE = _to_measure(MEAN_9.numpy())

def reward(x):
    METRIC = "Euclidean"  # "Wasserstein" or "Euclidean"
    imgs = x.detach().cpu().numpy().reshape(-1, IMG_SIZE, IMG_SIZE)
    flipped = imgs[:, ::-1, ::-1].reshape(-1, DIM)
    if METRIC == "Euclidean":
        dists = np.linalg.norm(flipped - MEAN_9.numpy(), axis=-1)
    else:
        dists = np.array([ot.emd2(_to_measure(row), _MEAN_9_MEASURE, _GROUND_COST) for row in flipped])
    return torch.tensor(-dists, dtype=x.dtype, device=x.device)
