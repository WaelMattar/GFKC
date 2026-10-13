import os
import gc
import time
import multiprocessing as mp
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion import GaussianModel, sample_FK
from distributions import MU_1, MU_2, gaussian_pdf, H1_VARIANTS, H1_RAW
from visualizations import normalize, kde_on_grid, grid_pts, gx, gy, GRID_RES

SEED             = 42
N_STEPS          = 100
XI               = 0     # 0: ODE sampling, 1: SDE sampling
RESAMPLING_TIMES = {}
H1_VARIANTS_TO_RUN = ["ratio", "symmetric"]
LINESTYLES = {"ratio": "-", "symmetric": "--"}

# Per-agent particle counts to sweep (total particles passed to sample_FK is 2x each value,
# matching main.py's convention: N_SAMPLE = per_agent_count * 2).
PER_AGENT_SIZES = [20_000, 35_000, 50_000, 75_000, 100_000, 150_000, 200_000, 500_000, 700_000, 850_000, 1_000_000, 1_200_000, 1_500_000, 1_700_000, 2_000_000]

os.makedirs('Figures', exist_ok=True)

eps = 1e-300


def run_sweep(h1_variant, num_threads):
    # The two variants are fully independent sweeps (different target, same models and
    # particle counts), so they run as separate processes in parallel -- true parallelism,
    # not just concurrency, since this is CPU-bound work the GIL would otherwise serialize.
    # num_threads caps each process's own thread pool so the two don't oversubscribe the CPU.
    torch.set_num_threads(num_threads)

    log_h1_fn, dlog_p_dlog_h1_fn = H1_VARIANTS[h1_variant]

    p1_grid  = gaussian_pdf(grid_pts.T, MU_1).reshape(GRID_RES, GRID_RES)
    p2_grid  = gaussian_pdf(grid_pts.T, MU_2).reshape(GRID_RES, GRID_RES)
    true_pdf = normalize(H1_RAW[h1_variant](p1_grid, p2_grid))

    l2_errors = []
    kl_divs   = []

    print(f"Convergence sweep -- H1_VARIANT = '{h1_variant}'")
    print("-" * 72)

    t_start = time.time()
    for idx, per_agent in enumerate(PER_AGENT_SIZES, 1):
        n_total = 2 * per_agent
        run_t0  = time.time()

        # Re-seed so every sweep point is an independent, reproducible experiment,
        # rather than one continuous RNG stream shared across differently-sized runs.
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model1 = GaussianModel(MU_1)
        model2 = GaussianModel(MU_2)

        samples, log_w, agent_ids, _ = sample_FK(
            [model1, model2],
            n                = n_total,
            n_steps          = N_STEPS,
            snapshot_times   = None,
            resampling_times = RESAMPLING_TIMES,
            log_h1           = log_h1_fn,
            dlog_p_dlog_h1   = dlog_p_dlog_h1_fn,
            xi               = XI,
        )
        log_w = log_w - log_w.max()
        w     = np.exp(log_w); w /= w.sum()
        ess   = 1.0 / (w ** 2).sum()

        combined = normalize(kde_on_grid(samples, w))
        error    = true_pdf - combined
        l2_err   = float(np.sqrt(np.trapezoid(np.trapezoid(error ** 2, gx, axis=1), gy)))
        kl       = float(np.trapezoid(np.trapezoid(
                       true_pdf * np.log(np.maximum(true_pdf, eps) / np.maximum(combined, eps)),
                       gx, axis=1), gy))

        l2_errors.append(l2_err)
        kl_divs.append(kl)

        elapsed = time.time() - run_t0
        print(f"  [{h1_variant}] [{idx}/{len(PER_AGENT_SIZES)}] per-agent = {per_agent:>9,d}  (n_total = {n_total:>9,d})  "
              f"ESS = {ess:>10.1f}  L2 err = {l2_err:.5f}  KL = {kl:.5f}  [{elapsed:.1f}s]")

        # Free the large per-run arrays before moving on to the next (possibly much larger) run.
        del samples, log_w, agent_ids, w, combined, error, model1, model2
        gc.collect()

    total_elapsed = time.time() - t_start
    print(f"[{h1_variant}] total sweep time: {total_elapsed:.1f}s")
    return h1_variant, l2_errors, kl_divs


if __name__ == '__main__':
    n_procs = len(H1_VARIANTS_TO_RUN)
    threads_per_proc = max(1, (os.cpu_count() or n_procs) // n_procs)

    t_start = time.time()
    with mp.Pool(processes=n_procs) as pool:
        results = pool.starmap(run_sweep, [(v, threads_per_proc) for v in H1_VARIANTS_TO_RUN])
    total_elapsed = time.time() - t_start
    print("-" * 72)
    print(f"Total wall-clock time (both variants, parallel): {total_elapsed:.1f}s")

    results_by_variant = {v: (l2, kl) for v, l2, kl in results}

    # --- plot: single axes, dual y-scale -- solid = ratio, dashed = symmetric,
    #     each of the 4 curves its own C0-C3 color.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    COLORS = {('ratio', 'l2'): 'C0', ('ratio', 'kl'): 'C1',
              ('symmetric', 'l2'): 'C2', ('symmetric', 'kl'): 'C3'}

    handles, labels = [], []
    for variant in H1_VARIANTS_TO_RUN:
        l2_errors, kl_divs = results_by_variant[variant]
        h_l2, = ax1.plot(PER_AGENT_SIZES, l2_errors, color=COLORS[(variant, 'l2')], marker='o',
                          linestyle=LINESTYLES[variant], label=f'L2 ({variant})')
        h_kl, = ax2.plot(PER_AGENT_SIZES, kl_divs, color=COLORS[(variant, 'kl')], marker='s',
                          linestyle=LINESTYLES[variant], label=f'KL ({variant})')
        handles += [h_l2, h_kl]
        labels  += [h_l2.get_label(), h_kl.get_label()]

    ax1.set_xscale('log')
    ax1.set_xlabel('particles per agent')
    ax1.set_ylabel(r'$\|h_1 - \widehat{h}_1\|_2$', color='black')
    ax1.tick_params(axis='y', color='black', labelcolor='black')
    ax2.set_ylabel(r'KL($h_1 \Vert \widehat{h}_1$)', color='black')
    ax2.tick_params(axis='y', color='black', labelcolor='black')

    ax1.legend(handles, labels, ncol=2, loc='upper right')

    fig.tight_layout()
    fig.savefig('Figures/gaussian_convergence.pdf', bbox_inches='tight')
    plt.show()
