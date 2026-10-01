import numpy as np
import torch

from distributions import log_h0, dlog_p_dlog_h0, A_0, A_1, A_0_prime, A_1_prime
from visualizations import plot_log_weights_per_agent


def _dlog_alpha_dt(t):
    return 1.0 / t.clamp(min=1e-6)

def _gamma(t):
    return 1.0 / t.clamp(min=1e-6)


class GaussianModel:

    def __init__(self, mu):
        self.mu     = torch.tensor(mu, dtype=torch.float32)
        self.device = torch.device('cpu')

    def score(self, t, x):
        mu = self.mu.to(x.device)
        return -(x - t * mu)


def _vector_field(t, x, score, xi=0.0):
    # xi=0: probability flow ODE; xi=1: full reverse SDE
    return _dlog_alpha_dt(t) * x + (1 + xi) * _gamma(t) * score


def _exact_div(v, x):
    div = torch.zeros(x.shape[0], device=x.device)
    for j in range(x.shape[1]):
        grad_j = torch.autograd.grad(v[:, j].sum(), x, create_graph=False, retain_graph=True)[0]
        div    = div + grad_j[:, j]
    return div


@torch.no_grad()
def project_to_final_time(model, x, t, dt, xi=0.0):
    x = x.detach()
    t = t.detach()
    n_inner = int(round(((1.0 - t.max()) / dt).item()))
    for _ in range(n_inner):
        s = model.score(t, x)
        v = _vector_field(t, x, s, xi=xi)
        g = _gamma(t)
        noise = (2 * xi * g * (1.0 - t) * dt).clamp(min=0).sqrt() * torch.randn_like(x)
        x = x + v * dt + noise
        t = (t + dt).clamp(max=1.0 - 1e-5)
    return x


def sample_individual(model, n, n_steps, xi=0.0):
    device = model.device
    dt = 1.0 / n_steps
    t  = torch.zeros(n, 1, device=device)
    x  = torch.randn(n, 2, device=device)
    for _ in range(n_steps):
        with torch.no_grad():
            s     = model.score(t, x)
            v     = _vector_field(t, x, s, xi=xi)
            g     = _gamma(t)
            noise = (2 * xi * g * (1.0 - t) * dt).clamp(min=0).sqrt() * torch.randn_like(x)
            x     = x + v * dt + noise
            t     = (t + dt).clamp(max=1.0 - 1e-5)
    return x.cpu().numpy()


def sample_FK(models, n, n_steps, snapshot_times, resampling_times, steering_times=None, *,
               log_h1, dlog_p_dlog_h1, reward_fn=None, kappa=1.0, xi=0.0):
    K      = len(models)
    device = models[0].device
    N_per  = n // K
    dt     = 1.0 / n_steps

    snap_step_map = {}
    snapshots     = []
    if snapshot_times is not None:
        for t_s in snapshot_times:
            snap_step_map[round(t_s / dt)] = t_s

    resample_step_map = {}
    if resampling_times is not None:
        for t_r in resampling_times:
            resample_step_map[round(t_r / dt)] = t_r
    if resample_step_map:
        header = "   t   |" + "".join(f" Agent {i+1:2d} |" for i in range(K)) + "   Total"
        sep    = "-" * len(header)

    steer_step_map = {}
    if steering_times is not None:
        for t_s in steering_times:
            steer_step_map[round(t_s / dt)] = t_s

    def log_gaussian_prior(x_val):
        return -np.log(2 * np.pi) - 0.5 * (x_val ** 2).sum(-1)

    x_list     = [torch.randn(N_per, 2, device=device) for _ in range(K)]
    t_list     = [torch.zeros(N_per, 1,  device=device) for _ in range(K)]
    log_w_list = [np.zeros(N_per)                       for _ in range(K)]
    log_p_list = [log_gaussian_prior(x_list[i]).unsqueeze(0).expand(K, -1).clone() for i in range(K)]

    if resample_step_map:
        print("\n--- Particle counts per agent ---")
        print(header); print(sep)
        print(f"  0.00 |" + "".join(f" {N_per:8d} |" for _ in range(K)) + f"   {N_per * K}")

    if 0 in snap_step_map:
        snapshots.append((snap_step_map[0], [(x_list[i].cpu().numpy().copy(), log_w_list[i].copy()) for i in range(K)]))

    for step in range(n_steps):
        current_step = step + 1
        for i in range(K):
            if len(log_w_list[i]) == 0:
                continue
            x_g = x_list[i].detach().requires_grad_(True)
            t_d = t_list[i].detach()

            scores = [m.score(t_d, x_g) for m in models]
            vfs    = [_vector_field(t_d, x_g, s, xi=xi) for s in scores]
            divs   = torch.stack([_exact_div(vf, x_g) for vf in vfs])

            # Precompute from log_p before any update in the j loop
            A_0_t      = A_0(t_d).squeeze(-1)
            A_1_t      = A_1(t_d).squeeze(-1)

            # Equation (A), steered only at the steering_times checkpoints:
            # q^steered = q + A_1_prime * r(final_estimate) / kappa ; otherwise plain q.
            lh0 = log_h0(log_p_list[i])
            lh1 = log_h1(log_p_list[i])
            q   = A_0_prime(t_d) * lh0 + A_1_prime(t_d) * lh1
            if current_step in steer_step_map:
                final_estimates = project_to_final_time(models[i], x_g, t_d, dt, xi=xi)
                q = q + A_1_prime(t_d) * reward_fn(final_estimates) / kappa

            # Differentiation of log_h0 and log_h1 with respect to log_p
            lglh0 = dlog_p_dlog_h0(log_p_list[i])
            lglh1 = dlog_p_dlog_h1(log_p_list[i])

            sum_u_r = torch.zeros_like(divs[0])
            for j in range(K):
                r_j = A_0_t * lglh0[j] + A_1_t * lglh1[j]                            # Eq (B)
                u_ij = divs[j] + ((vfs[j] - vfs[i]) * scores[j]).sum(-1)             # Eq (C)
                sum_u_r += u_ij.detach() * r_j
                log_p_list[i][j] = (log_p_list[i][j] - u_ij.detach() * dt).detach()  # Eq (D)

            g_i = divs[i] + q - sum_u_r                                # Eq (E)
            log_w_list[i] += g_i.detach().cpu().numpy() * dt           # Eq (F)

            with torch.no_grad():                                      # Eq (G)
                g     = _gamma(t_d)
                noise = (2 * xi * g * (1.0 - t_d) * dt).clamp(min=0).sqrt() * torch.randn_like(x_list[i])
                x_list[i] = x_list[i].detach() + vfs[i].detach() * dt + noise
                t_list[i] = (t_list[i] + dt).clamp(max=1.0 - 1e-5)

        if current_step in snap_step_map:
            snapshots.append((snap_step_map[current_step],
                              [(x_list[i].cpu().numpy().copy(), log_w_list[i].copy())
                               for i in range(K) if len(log_w_list[i]) > 0]))

        # Resampling block
        if current_step in resample_step_map:
            t_r         = resample_step_map[current_step]
            plot_log_weights_per_agent(log_w_list, label=f't = {t_r:.2f}')
            all_x_pool  = torch.cat([x_list[i]     for i in range(K)], dim=0)
            all_lp_pool = torch.cat([log_p_list[i] for i in range(K)], dim=1)
            all_t_pool  = torch.cat([t_list[i]     for i in range(K)], dim=0)
            all_lw_pool = np.concatenate([log_w_list[i] for i in range(K)])
            all_aids    = np.concatenate([np.full(len(log_w_list[i]), i, dtype=int) for i in range(K)])
            lw_shifted  = all_lw_pool - all_lw_pool.max()
            w_norm      = np.exp(lw_shifted); w_norm /= w_norm.sum()
            N_tot       = len(all_lw_pool)
            chosen      = np.random.choice(N_tot, size=N_tot, replace=True, p=w_norm)
            chosen_t    = torch.tensor(chosen, device=device)
            new_x       = all_x_pool[chosen_t].detach()
            new_lp      = all_lp_pool[:, chosen_t].detach()
            new_t_tens  = all_t_pool[chosen_t].detach()
            new_aids    = all_aids[chosen]
            for i in range(K):
                mask_i         = new_aids == i
                x_list[i]     = new_x[mask_i].detach()
                log_p_list[i] = new_lp[:, mask_i].detach()
                t_list[i]     = new_t_tens[mask_i].detach()
                log_w_list[i] = np.zeros(int(mask_i.sum()))
            counts = [len(log_w_list[i]) for i in range(K)]
            print(f"  {t_r:.2f} |" + "".join(f" {c:8d} |" for c in counts) + f"   {sum(counts)}")

    if resample_step_map:
        final_counts = [len(log_w_list[i]) for i in range(K)]
        print(f"  1.00 |" + "".join(f" {c:8d} |" for c in final_counts) + f"   {sum(final_counts)}")
        print(sep)

    all_x     = np.vstack([x_list[i].cpu().numpy() for i in range(K)])
    all_log_w = np.concatenate([log_w_list[i] for i in range(K)])
    agent_ids = np.concatenate([np.full(len(log_w_list[i]), i, dtype=int) for i in range(K)])
    snapshots.sort(key=lambda s: s[0])
    return all_x, all_log_w, agent_ids, snapshots
