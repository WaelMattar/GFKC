import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import gaussian_kde

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12

def _to_img(arr, img_size=28):
    """Flat array in [-1, 1] → (img_size, img_size) image in [0, 1]."""
    return ((arr.reshape(img_size, img_size) + 1.0) / 2.0).clip(0, 1)

def plot_agent_grid(samples, title, save_path, img_size=28, nrows=4, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    for ax, img in zip(axes.ravel(), samples):
        ax.imshow(_to_img(img, img_size), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def plot_all_particles(samples, log_weights, agent_ids, rewards, save_path='Figures/all_particles.pdf', img_size=28, nrows=10, ncols=10):
    from matplotlib.colors import LinearSegmentedColormap
    AGENT_CMAPS = [
        LinearSegmentedColormap.from_list('black_cyan',  ['black', 'cyan']),
        LinearSegmentedColormap.from_list('black_magenta', ['black', 'magenta']),
    ]
    sorted_idx = np.argsort(log_weights)[::-1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    for ax, idx in zip(axes.ravel(), sorted_idx):
        ax.imshow(_to_img(samples[idx], img_size), cmap=AGENT_CMAPS[agent_ids[idx]], vmin=0, vmax=1)
        # ax.set_title(f'log_w={log_weights[idx]:.2f}\nr={rewards[idx]:.2f}', fontsize=6)
        ax.axis('off')
    # fig.suptitle(
    #     r'All particles sorted by weights'
    #     '\n' + r'cyan = Agent 1   magenta = Agent 2',
    #     fontsize=11,
    # )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def plot_log_w_hist(log_weights, agent_ids, save_path='Figures/log_w_hist.pdf'):
    fig, ax = plt.subplots()
    AGENT_COLORS = ['cyan', 'magenta']
    groups = [log_weights[agent_ids == 0], log_weights[agent_ids == 1]]
    bin_edges = np.histogram_bin_edges(log_weights, bins=len(log_weights))
    bin_width = bin_edges[1] - bin_edges[0]

    FORCE_THRESHOLD = 10
    FORCE_FRACTION = 0.1

    # Natural per-agent bar heights under plain fraction-of-count normalization (weight = 1/n).
    counts_n_peak = []
    for data in groups:
        n = len(data)
        peak = np.histogram(data, bins=bin_edges)[0].max() / n if n > 0 else 0.0
        counts_n_peak.append((n, peak))

    # FIGURE_HEIGHT is set by the well-sampled agent(s); a sparse agent's bars are then
    # forcibly rescaled so its own tallest bar sits at FORCE_FRACTION of that height,
    # instead of towering over (or vanishing next to) the well-sampled agent's histogram.
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
            # Too few samples for a data-driven bandwidth (n=1 has no variance at all) --
            # place a fixed-width Gaussian per sample instead, then rescale it by the
            # same forced factor as this agent's bars, to stay visually consistent.
            density = np.zeros_like(xs)
            for mu in data:
                density += np.exp(-0.5 * ((xs - mu) / STD_FALLBACK) ** 2) / (STD_FALLBACK * np.sqrt(2 * np.pi))
            density /= n
        # Scale the density by bin_width to match the bar heights' fraction-per-bin scale.
        ax.plot(xs, density * bin_width * scale, color=color, linewidth=3)
    # ax.set_title('Histogram of log weights')
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def plot_sharpening_sum_effect(results, save_path='Figures/sharpening_sum_effect.pdf'):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    for ax, (pow_value, log_weights, agent_ids) in zip(axes, results):
        first_agent_log_weights = log_weights[agent_ids == 0]
        second_agent_log_weights = log_weights[agent_ids == 1]
        ax.hist([first_agent_log_weights, second_agent_log_weights],
                bins=len(log_weights),
                color=['skyblue', 'magenta'],
                label=['first agent', 'second agent'],
                edgecolor='black',
                linewidth=0.2)
        ax.set_xlabel(f'pow = {pow_value:g}', fontsize=26)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def plot_log_w_vs_reward(log_weights, rewards, agent_ids, save_path='Figures/log_w_vs_reward.pdf'):
    AGENT_COLORS = ['cyan', 'magenta']
    fig, ax = plt.subplots()
    for k, label in enumerate(['Agent 1', 'Agent 2']):
        mask = agent_ids == k
        ax.scatter(log_weights[mask], rewards[mask], s=50, color=AGENT_COLORS[k],
                   edgecolors='black', linewidths=0.5, label=label)
    # slope, intercept = np.polyfit(log_weights, rewards, 1)
    # x_line = np.array([log_weights.min(), log_weights.max()])
    # y_line = slope * x_line + intercept
    # y_pred = slope * log_weights + intercept
    # ss_res = np.sum((rewards - y_pred) ** 2)
    # ss_tot = np.sum((rewards - rewards.mean()) ** 2)
    # r2 = 1 - ss_res / ss_tot
    # ax.plot(x_line, y_line, color='red', linewidth=2, label=f'$R^2$ = {r2:.3f}')
    ax.set_xlabel('log weights', fontsize=16)
    ax.set_ylabel('reward value', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def plot_best_vs_reference(best_image, reference_image, save_path='Figures/best_vs_reference.pdf', img_size=28):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    axes[0].imshow(_to_img(best_image, img_size), cmap='gray', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[1].imshow(_to_img(reference_image, img_size), cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def plot_reward_hist(rewards, agent_ids, save_path='Figures/reward_hist.pdf'):
    fig = plt.figure()
    first_agent_rewards = rewards[agent_ids == 0]
    second_agent_rewards = rewards[agent_ids == 1]
    plt.hist([first_agent_rewards, second_agent_rewards],
             bins=len(rewards),
             color=['skyblue', 'magenta'],
             label=['first agent', 'second agent'],
             edgecolor='black',
             linewidth=0.2)
    plt.title('Histogram of reward values')
    plt.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
