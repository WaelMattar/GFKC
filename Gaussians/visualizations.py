import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

mpl.rcParams['font.family']      = 'serif'
mpl.rcParams['font.serif']       = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size']        = 16
mpl.rcParams['axes.titlesize']   = 16
mpl.rcParams['axes.labelsize']   = 16
mpl.rcParams['xtick.labelsize']  = 16
mpl.rcParams['ytick.labelsize']  = 16

BANDWIDTH_FACTOR = 1.0
GX_LO, GX_HI    = 0.0, 7.0
GY_LO, GY_HI    = 0.0, 7.0
GRID_RES         = 80

gx       = np.linspace(GX_LO, GX_HI, GRID_RES)
gy       = np.linspace(GY_LO, GY_HI, GRID_RES)
Gx, Gy   = np.meshgrid(gx, gy)
grid_pts = np.vstack([Gx.ravel(), Gy.ravel()])


def kde_on_grid(samples, weights):
    bw  = BANDWIDTH_FACTOR * len(samples) ** (-1.0 / (2 + 4))
    kde = gaussian_kde(samples.T, bw_method=bw, weights=weights)
    return kde(grid_pts).reshape(GRID_RES, GRID_RES)

def normalize(Z):
    return Z / np.trapezoid(np.trapezoid(Z, gx, axis=1), gy)


_TICKS = np.arange(int(np.ceil(GX_LO)), int(np.floor(GX_HI)) + 1)

def _style_ax(ax):
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim(GX_LO, GX_HI); ax.set_ylim(GY_LO, GY_HI)
    ax.set_xticks(_TICKS); ax.set_yticks(_TICKS)
    ax.set_aspect('equal'); ax.tick_params(axis='both')


def plot_comparison(combined, true_pdf, h1_label=r'$h_1(x)$'):
    error = np.abs(true_pdf - combined)
    fig, (ax_p, ax_q, ax_e) = plt.subplots(1, 3, figsize=(16, 5))
    ax_p.contourf(Gx, Gy, true_pdf, levels=30, cmap='viridis')
    # ax_p.set_title('True $h_1(x)$')
    _style_ax(ax_p)
    ax_q.contourf(Gx, Gy, combined, levels=30, cmap='viridis')
    # ax_q.set_title('Recovered $q(x)$')
    _style_ax(ax_q)
    err_plot = ax_e.contourf(Gx, Gy, error, levels=30, cmap='magma')
    # ax_e.set_title(r'$|$True $-$ Recovered$|$')
    _style_ax(ax_e)
    fig.colorbar(err_plot, ax=ax_e, fraction=0.046, pad=0.04)
    # fig.suptitle(f'Recovery of {h1_label}  via FK reweighting')
    fig.tight_layout()
    fig.savefig('Figures/gaussian_comparison.pdf', bbox_inches='tight')


def plot_individual_models(ind1, ind2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.contourf(Gx, Gy, ind1, levels=30, cmap='viridis')
    ax1.set_title(r'$\hat{p}_1(x)$  — agent 1'); _style_ax(ax1)
    ax2.contourf(Gx, Gy, ind2, levels=30, cmap='viridis')
    ax2.set_title(r'$\hat{p}_2(x)$  — agent 2'); _style_ax(ax2)
    fig.tight_layout()
    fig.savefig('Figures/gaussian_individual_models.pdf', bbox_inches='tight')


def plot_log_weights_per_agent(log_w_list, label, bins=80):
    K      = len(log_w_list)
    colors = [f'C{i}' for i in range(K)]
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))
    for i, ax in enumerate(axes):
        lw = log_w_list[i]
        ax.hist(lw, bins=bins, color=colors[i], alpha=0.8)
        ax.set_title(f'Agent {i + 1}')
        ax.set_xlabel('log weight')
        ax.set_ylabel('count')
        ax.legend([f'n = {len(lw):,}'], loc='upper left')
    fig.suptitle(f'Log-weights — {label}')
    fig.tight_layout()
    safe = label.replace(' ', '_').replace('=', '').replace('.', 'p')
    fig.savefig(f'Figures/gaussian_logw_{safe}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_unweighted_kde_agents(samples, agent_ids, log_w, n_scatter=30000):
    K      = len(np.unique(agent_ids))
    fig, axes = plt.subplots(2, K, figsize=(5 * K, 10))

    cmap   = mpl.colormaps['Blues']
    lw_all = log_w - log_w.max()

    for i in range(K):
        mask = agent_ids == i
        s_i  = samples[mask]

        w_uni = np.ones(mask.sum()) / mask.sum()
        dens  = normalize(kde_on_grid(s_i, w_uni))
        ax_top = axes[0, i]
        ax_top.contourf(Gx, Gy, dens, levels=30, cmap='viridis')
        ax_top.set_title(f'Agent {i + 1} — unweighted KDE'); _style_ax(ax_top)

        # Per-agent independent color scale
        lw_i   = lw_all[mask]
        norm_i = mpl.colors.Normalize(vmin=lw_i.min(), vmax=lw_i.max())

        n_pts = min(n_scatter, len(s_i))
        idx   = np.random.choice(len(s_i), size=n_pts, replace=False)

        ax_bot = axes[1, i]
        ax_bot.set_facecolor('white')
        sc = ax_bot.scatter(s_i[idx, 0], s_i[idx, 1], s=4,
                            c=lw_i[idx], cmap=cmap, norm=norm_i, edgecolors='face')
        ax_bot.set_title(f'Agent {i + 1} — FK-weighted scatter'); _style_ax(ax_bot)

        # One colorbar per scatter subplot, placed to its right
        fig.colorbar(sc, ax=ax_bot, fraction=0.046, pad=0.04, label='log FK weight')

    axes[0, 0].set_ylabel('Unweighted KDE', fontsize=12)
    axes[1, 0].set_ylabel('FK-weighted scatter', fontsize=12)

    fig.suptitle('Unweighted KDE and FK-weighted particle scatter per agent')
    fig.tight_layout()
    fig.savefig('Figures/gaussian_unweighted_kde_agents.pdf', bbox_inches='tight')


def plot_agent_evolution(snapshot_densities):
    K     = max(len(densities_k) for _, densities_k in snapshot_densities)
    ncols = len(snapshot_densities)
    fig, axes = plt.subplots(K, ncols, figsize=(5 * ncols, 5 * K))
    axes = np.array(axes).reshape(K, ncols)
    for col, (t_val, densities_k) in enumerate(snapshot_densities):
        for row in range(K):
            ax = axes[row, col]
            if row >= len(densities_k):
                ax.set_visible(False)
                continue
            ax.contourf(Gx, Gy, densities_k[row], levels=30, cmap='viridis')
            ax.set_xlim(GX_LO, GX_HI); ax.set_ylim(GY_LO, GY_HI)
            ax.set_xticks(_TICKS); ax.set_yticks(_TICKS)
            ax.set_aspect('equal'); ax.tick_params(axis='both', labelsize=10)
            if row == 0:
                ax.set_title(f'$t = {t_val:.2f}$')
            if col == 0:
                ax.set_ylabel(f'Agent {row + 1}', fontsize=12)
    fig.tight_layout()
    fig.savefig('Figures/gaussian_agent_evolution.pdf', bbox_inches='tight')
