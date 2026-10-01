import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from distributions import DIM, log_h0, dlog_p_dlog_h0, A_0, A_1, A_0_prime, A_1_prime

BETA_0 = 0.1
BETA_1 = 20.0
N_STEPS = 500

def _log_alpha(t):
    u = 1.0 - t
    return -0.5 * BETA_0 * u - 0.25 * (BETA_1 - BETA_0) * u ** 2

def _dlog_alpha_dt(t):
    u = 1.0 - t
    return 0.5 * BETA_0 + 0.5 * (BETA_1 - BETA_0) * u

def _sigma(t):
    return (1.0 - torch.exp(2.0 * _log_alpha(t))).clamp(min=1e-8).sqrt()

def _vector_field(t, x, score, xi=0.0):
    grad_log_p = score / _sigma(t)
    return _dlog_alpha_dt(t) * (x + (1 + xi) * grad_log_p)


class _SinEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1))
        args = t.view(-1, 1) * freqs[None, :] * 1000.0
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, dim=DIM, hidden=2048, time_dim=64):
        super().__init__()
        self.time_emb = nn.Sequential(
            _SinEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),         nn.SiLU(),
            nn.Linear(hidden, hidden),         nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, x):
        t_emb = self.time_emb(t.view(-1))
        return self.net(torch.cat([x, t_emb], dim=-1))


class DiffusionModel:
    def __init__(self, device='cpu', dim=DIM, hidden=2048):
        self.device = torch.device(device)
        self.dim = dim
        self.net = ScoreNet(dim=dim, hidden=hidden).to(self.device)

    def train(self, data, n_iters=10000, batch_size=512, lr=2e-4, print_every=5000):
        data = data.to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        pbar = tqdm(range(1, n_iters + 1), desc="training")
        for it in pbar:
            idx = torch.randint(len(data), (batch_size,), device=self.device)
            x0 = data[idx]
            t = torch.rand(batch_size, 1, device=self.device).clamp(1e-3, 1.0 - 1e-3)
            eps = torch.randn_like(x0)
            xt = torch.exp(_log_alpha(t)) * x0 + _sigma(t) * eps
            loss = ((eps + self.net(1.0 - t, xt)) ** 2).mean()  # net is parametrized in u = 1 - t
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % print_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def score(self, t, x):
        return self.net(1.0 - t, x)

    @torch.no_grad()
    def sample(self, n, n_steps=N_STEPS, xi=0.0):
        self.net.eval()
        dt = 1.0 / n_steps
        t = torch.zeros(n, 1, device=self.device)          # t = 0  ->  pure noise
        x = torch.randn(n, self.dim, device=self.device)

        for step in range(1, n_steps + 1):
            s = self.score(t, x)
            v = _vector_field(t, x, s, xi=xi)
            noise = (2 * xi * _dlog_alpha_dt(t) * (1.0 - t) * dt).clamp(min=0).sqrt() * torch.randn_like(x)
            x = x + v * dt + noise
            t = (t + dt).clamp(max=1.0 - 1e-5)

        return x.cpu().numpy()


def _log_gauss(x):
    return -0.5 * x.shape[-1] * np.log(2 * np.pi) - 0.5 * (x ** 2).sum(-1)


def _hutchinson_div(v, x, n_probes=1):
    estimates = []
    for _ in range(n_probes):
        eps = torch.randn_like(x)
        vjp = torch.autograd.grad(v, x, grad_outputs=eps, create_graph=False, retain_graph=True)[0]
        estimates.append((eps * vjp).sum(-1))
    return torch.stack(estimates).mean(0)


@torch.no_grad()
def project_to_final_time(model, x, t, dt, xi=0.0):
    x = x.detach()
    t = t.detach()
    n_inner = int(round(((1.0 - t.max()) / dt).item()))
    for _ in range(n_inner):
        s = model.score(t, x)
        v = _vector_field(t, x, s, xi=xi)
        noise = (2 * xi * _dlog_alpha_dt(t) * (1.0 - t) * dt).clamp(min=0).sqrt() * torch.randn_like(x)
        x = x + v * dt + noise
        t = (t + dt).clamp(max=1.0 - 1e-5)
    return x


def scale_to_range(x, lo=-30.0, hi=30.0):
    x_min, x_max = x.min(), x.max()
    assert x_min <= x_max, "Error in scaling!"
    # avoid divide-by-zero if all values are equal
    if x_max == x_min:
        return torch.full_like(x, (lo + hi) / 2)
    return (x - x_min) / (x_max - x_min) * (hi - lo) + lo


def sample_FK(models, n, n_steps, n_probes, resampling_times, steering_times=None, *,
               log_h1, dlog_p_dlog_h1, reward_fn=None, kappa=1.0, xi=0.0):
    K = len(models)
    device = models[0].device
    dim = models[0].dim
    N_per = n // K
    dt = 1.0 / n_steps

    for m in models:
        m.net.eval()
        m.net.double()

    resample_step_map = {}
    if resampling_times:
        for t_r in resampling_times:
            resample_step_map[round(t_r / dt)] = t_r
    if resample_step_map:
        header = "   t   |" + "".join(f" Agent {i+1:2d} |" for i in range(K)) + "   Total"
        sep = "-" * len(header)

    steer_step_map = {}
    if steering_times:
        for t_s in steering_times:
            steer_step_map[round(t_s / dt)] = t_s

    x_list = [torch.randn(N_per, dim, device=device, dtype=torch.float64) for _ in range(K)]
    t_list = [torch.zeros(N_per, 1, device=device, dtype=torch.float64) for _ in range(K)]
    log_w_list = [np.zeros(N_per) for _ in range(K)]
    log_p_list = [_log_gauss(x_list[i]).unsqueeze(0).expand(K, -1).clone() for i in range(K)]

    if resample_step_map:
        print("\n--- Particle counts per agent ---")
        print(header)
        print(sep)
        print(f"  0.00 |" + "".join(f" {N_per:8d} |" for _ in range(K)) + f"   {N_per * K}")

    for step in range(n_steps):
        current_step = step + 1

        # Preparation for numerical stability: clamping log
        for k in range(K):
            log_p_list[k] = log_p_list[k].clamp(max=700)

        for i in range(K):
            if len(log_w_list[i]) == 0:
                continue
            x_g = x_list[i].detach().requires_grad_(True)
            t_d = t_list[i].detach()

            scores = [m.score(t_d, x_g) for m in models]
            vfs = [_vector_field(t_d, x_g, s, xi=xi) for s in scores]
            divs = torch.stack([_hutchinson_div(vf, x_g, n_probes) for vf in vfs])

            # Precompute the interpolating coefficients at time t_d
            A_0_t = A_0(t_d).squeeze(-1)
            A_1_t = A_1(t_d).squeeze(-1)

            # Equation (A), steered only at the steering_times checkpoints:
            # q^steered = q + A_1_prime * r(final_estimate) / kappa ; otherwise plain q.
            lh0 = log_h0(log_p_list[i])
            lh1 = log_h1(log_p_list[i])
            q = A_0_prime(t_d) * lh0 + A_1_prime(t_d) * lh1
            if current_step in steer_step_map:
                final_estimates = project_to_final_time(models[i], x_g, t_d, dt, xi=xi)
                q = q + A_1_prime(t_d) * reward_fn(final_estimates) / kappa

            # Differentiation of log_h0 and log_h1 with respect to log_p
            lglh0 = dlog_p_dlog_h0(log_p_list[i])
            lglh1 = dlog_p_dlog_h1(log_p_list[i])

            sum_u_r = torch.zeros_like(divs[0])
            for j in range(K):
                r_j = A_0_t * lglh0[j] + A_1_t * lglh1[j]  # Equation (B)
                grad_log_p_j = scores[j] / _sigma(t_d)
                u_ij = divs[j] + ((vfs[j] - vfs[i]) * grad_log_p_j).sum(-1)  # Equation (C)
                sum_u_r += u_ij.detach() * r_j
                log_p_list[i][j] = (log_p_list[i][j] - u_ij.detach() * dt).detach()  # Equation (D)

            g_i = divs[i] + q - sum_u_r  # Equation (E)
            log_w_list[i] += g_i.detach().cpu().numpy() * dt  # Equation (F)

            with torch.no_grad():# Equation (G)
                noise = (2 * xi * _dlog_alpha_dt(t_d) * (1.0 - t_d) * dt).clamp(min=0).sqrt() * torch.randn_like(x_list[i])
                x_list[i] = x_list[i].detach() + vfs[i].detach() * dt + noise
                t_list[i] = (t_list[i] + dt).clamp(max=1.0 - 1e-5)

        # Preparation for numerical stability: shifting log weights
        mean_log_weight = np.concatenate(log_w_list).mean()
        for k in range(K):
            log_w_list[k] = log_w_list[k] - mean_log_weight

        # Resampling block
        if current_step in resample_step_map:
            t_r = resample_step_map[current_step]
            all_x_pool = torch.cat(x_list, dim=0)
            all_lp_pool = torch.cat(log_p_list, dim=1)
            all_t_pool = torch.cat(t_list, dim=0)
            all_lw_pool = np.concatenate(log_w_list)
            all_aids = np.concatenate([np.full(len(log_w_list[i]), i, dtype=int) for i in range(K)])
            N_tot = len(all_lw_pool)
            if not np.all(np.isfinite(all_lw_pool)):
                print(f"  Warning: non-finite log weights at t={t_r:.2f} — falling back to uniform resampling")
                w_norm = np.ones(N_tot) / N_tot
            else:
                lw_shifted = all_lw_pool - all_lw_pool.max()
                w_norm = np.exp(lw_shifted)
                w_norm /= w_norm.sum()
            chosen = np.random.choice(N_tot, size=N_tot, replace=True, p=w_norm)
            chosen_t = torch.tensor(chosen, device=device)
            new_x = all_x_pool[chosen_t].detach()
            new_lp = all_lp_pool[:, chosen_t].detach()
            new_t_tens = all_t_pool[chosen_t].detach()
            new_aids = all_aids[chosen]
            for i in range(K):
                mask_i = new_aids == i
                x_list[i] = new_x[mask_i].detach()
                log_p_list[i] = new_lp[:, mask_i].detach()
                t_list[i] = new_t_tens[mask_i].detach()
                log_w_list[i] = np.zeros(int(mask_i.sum()))
            counts = [len(log_w_list[i]) for i in range(K)]
            print(f"  {t_r:.2f} |" + "".join(f" {c:8d} |" for c in counts) + f"   {sum(counts)}")

    if resample_step_map:
        final_counts = [len(log_w_list[i]) for i in range(K)]
        print(f"  1.00 |" + "".join(f" {c:8d} |" for c in final_counts) + f"   {sum(final_counts)}")
        print(sep)

    all_x = np.vstack([x_list[i].cpu().numpy() for i in range(K)])
    all_log_w = np.concatenate(log_w_list)
    agent_ids = np.concatenate([np.full(len(log_w_list[i]), i, dtype=int) for i in range(K)])
    return all_x, all_log_w, agent_ids, log_p_list
