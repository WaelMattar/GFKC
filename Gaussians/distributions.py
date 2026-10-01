import math
import numpy as np
import torch
import torch.nn.functional as F

MU_1 = np.array([3.0, 3.0])
MU_2 = np.array([3.5, 3.5])
C = 100.0
LOG_C = math.log(C)

def gaussian_pdf(pts, mu):
    """N(mu, I) PDF evaluated at pts — pts is (N, D) array."""
    d = mu.shape[0]
    diff = pts - mu
    return np.exp(-0.5 * (diff ** 2).sum(-1)) / (2 * np.pi) ** (d / 2)


def log_h0(log_p):
    # h0 = geometric mean sqrt(p_1 * p_2)  ->  log = mean(log_p)
    return log_p.mean(dim=0)

def dlog_p_dlog_h0(log_p):
    # ∂(log h_0)/∂(log p_1) = ∂(log h_0)/∂(log p_2) = 1/2
    return torch.full(size=log_p.size(), fill_value=1/2)

def log_h1_ratio(log_p):
    # h1 = p_1 / (1 + C*p_2)
    return log_p[0] - F.softplus(log_p[1] + LOG_C)

def dlog_p_dlog_h1_ratio(log_p):
    # ∂(log h_1)/∂(log p_1) = 1
    # ∂(log h_1)/∂(log p_2) = - C*p_2 / (1 + C*p_2)
    d0 = torch.ones_like(log_p[0:1])
    d1 = -torch.sigmoid(log_p[1:2] + LOG_C)
    return torch.cat([d0, d1], dim=0)

def log_h1_symmetric(log_p):
    # h1 = 1/2 * ( p_1/(1+C*p_2) + p_2/(1+C*p_1) )
    log_A = log_p[0] - F.softplus(log_p[1] + LOG_C)
    log_B = log_p[1] - F.softplus(log_p[0] + LOG_C)
    return torch.logaddexp(log_A, log_B) - math.log(2.0)

def dlog_p_dlog_h1_symmetric(log_p):
    # the constant 1/2 factor drops out of the log-derivative
    log_A = log_p[0:1] - F.softplus(log_p[1:2] + LOG_C)
    log_B = log_p[1:2] - F.softplus(log_p[0:1] + LOG_C)
    w_A = torch.sigmoid(log_A - log_B)
    w_B = torch.sigmoid(log_B - log_A)
    d0 = w_A - w_B * torch.sigmoid(log_p[0:1] + LOG_C)
    d1 = w_B - w_A * torch.sigmoid(log_p[1:2] + LOG_C)
    return torch.cat([d0, d1], dim=0)

H1_VARIANTS = {
    "ratio":     (log_h1_ratio,     dlog_p_dlog_h1_ratio),
    "symmetric": (log_h1_symmetric, dlog_p_dlog_h1_symmetric),
}
H1_RAW = {
    "ratio":     lambda p1, p2: p1 / (1 + C * p2),
    "symmetric": lambda p1, p2: 0.5 * (p1 / (1 + C * p2) + p2 / (1 + C * p1)),
}
H1_LABELS = {
    "ratio":     r'$h_1(x)=p_1(x)\,/\,(1+C\,p_2(x))$',
    "symmetric": r'$h_1(x)=\frac{1}{2}\left(\frac{p_1(x)}{1+C\,p_2(x)}+\frac{p_2(x)}{1+C\,p_1(x)}\right)$',
}

# Feynman-Kac anneal along the diffusion time (t = 0 noise, t = 1 data):
def A_0(t_val): return 1.0 - t_val
def A_1(t_val): return t_val
def A_0_prime(t_val): return -1.0
def A_1_prime(t_val): return 1.0

# Steering: interpolate from log(h0) to log(h1 * e^{r(x)/KAPPA}) instead of log(h1).
def reward(x):
    return -x.norm(dim=-1)
