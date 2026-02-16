import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

CMU_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Ablation axes and their values ──
ABLATIONS = {
    'T':           [500, 1000, 2000, 5000],
    'num_iter':    [10, 20, 50, 100],
    'rho':         [0.9, 0.95, 0.99],
    'queue_class': [5, 10, 15, 20],
}
BASELINE = {'num_iter': 20, 'rho': 0.95, 'T': 1000, 'queue_class': 10}

ALPHAS = ['0.01', '0.1', '0.5', '1.0']
GAPS = ['1', '0.5', '0.05', '0.01']  # will be plotted left-to-right then x-inverted
GAPS_FLOAT = [1, 0.5, 0.05, 0.01]

AXIS_LABELS = {
    'T': 'Horizon T',
    'num_iter': 'Gradient Steps',
    'rho': r'Traffic Intensity $\rho$',
    'queue_class': 'Queue Classes',
}
AXIS_VAL_FMT = {
    'T':           lambda v: f'T={v}',
    'num_iter':    lambda v: f'K={v}',
    'rho':         lambda v: f'ρ={v}',
    'queue_class': lambda v: f'n={v}',
}


def load_ablation(method, axis, val):
    path = os.path.join(CMU_DIR, f'{method}_ablation_{axis}_{val}.json')
    with open(path, 'r') as f:
        return json.load(f)


def get_costs_per_gap(data):
    """Return {gap_str: array of avg_costs} averaged over all alphas."""
    by_gap = {}
    for gap in GAPS:
        costs = []
        for alpha in ALPHAS:
            if alpha in data and gap in data[alpha]:
                for run in data[alpha][gap]:
                    costs.append(run['avg_cost'])
        by_gap[gap] = np.array(costs)
    return by_gap


# ─────────────────────────────────────────────────────────────────
# Figure 1 (main): 2×2 grid, Fig 9.2 style per ablation axis
#   X-axis: gap ε (decreasing → right to left via invert)
#   Y-axis: avg holding cost
#   Lines : one per ablation value, solid = Pathwise, dashed = REINFORCE
#   Error bars: ±1 std dev across 100 runs × 4 alphas
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (axis_name, axis_values) in enumerate(ABLATIONS.items()):
    ax = axes[idx]
    n_vals = len(axis_values)
    colors = cm.viridis(np.linspace(0.15, 0.85, n_vals))

    x_pos = np.arange(len(GAPS))  # categorical positions for gap values

    for vi, val in enumerate(axis_values):
        pw_data = load_ablation('pathwise', axis_name, val)
        rf_data = load_ablation('reinforce', axis_name, val)
        pw_by_gap = get_costs_per_gap(pw_data)
        rf_by_gap = get_costs_per_gap(rf_data)

        pw_means = [np.mean(pw_by_gap[g]) for g in GAPS]
        pw_std = [1.96 * np.std(pw_by_gap[g]) / np.sqrt(len(pw_by_gap[g])) for g in GAPS]
        rf_means = [np.mean(rf_by_gap[g]) for g in GAPS]
        rf_std = [1.96 * np.std(rf_by_gap[g]) / np.sqrt(len(rf_by_gap[g])) for g in GAPS]

        lbl = AXIS_VAL_FMT[axis_name](val)
        is_baseline = (val == BASELINE[axis_name])
        lw_pw = 3.0 if is_baseline else 2.0
        lw_rf = 2.5 if is_baseline else 1.5

        ax.errorbar(x_pos, pw_means, yerr=pw_std, marker='s', linestyle='-',
                    linewidth=lw_pw, capsize=4, color=colors[vi],
                    label=f'PW: {lbl}' + (' *' if is_baseline else ''))
        ax.errorbar(x_pos, rf_means, yerr=rf_std, marker='o', linestyle='--',
                    linewidth=lw_rf, capsize=4, color=colors[vi], alpha=0.75,
                    label=f'RF: {lbl}' + (' *' if is_baseline else ''))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=12)
    ax.set_ylabel('Avg Holding Cost', fontsize=12)
    ax.set_title(f'Ablation: {AXIS_LABELS[axis_name]}', fontsize=13)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle(
    'Ablation Analysis — Pathwise (solid) vs REINFORCE (dashed)\n'
    '(* = baseline setting; error bars = 95% CI)',
    fontsize=14, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'ablation_eps_vs_cost.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved ablation_eps_vs_cost.png')


# ─────────────────────────────────────────────────────────────────
# Figure 2: Same layout but showing cost RATIO (Pathwise / REINFORCE)
#   Highlights where one method dominates across ε and ablation settings
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (axis_name, axis_values) in enumerate(ABLATIONS.items()):
    ax = axes[idx]
    n_vals = len(axis_values)
    colors = cm.viridis(np.linspace(0.15, 0.85, n_vals))
    x_pos = np.arange(len(GAPS))

    for vi, val in enumerate(axis_values):
        pw_data = load_ablation('pathwise', axis_name, val)
        rf_data = load_ablation('reinforce', axis_name, val)
        pw_by_gap = get_costs_per_gap(pw_data)
        rf_by_gap = get_costs_per_gap(rf_data)

        ratios = [np.mean(pw_by_gap[g]) / np.mean(rf_by_gap[g]) for g in GAPS]
        lbl = AXIS_VAL_FMT[axis_name](val)
        is_baseline = (val == BASELINE[axis_name])
        lw = 3.0 if is_baseline else 2.0

        ax.plot(x_pos, ratios, marker='s', linewidth=lw, color=colors[vi],
                label=lbl + (' *' if is_baseline else ''))

    ax.axhline(1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=12)
    ax.set_ylabel('Cost Ratio (PW / RF)', fontsize=12)
    ax.set_title(f'Ablation: {AXIS_LABELS[axis_name]}', fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    'Relative Performance: Pathwise / REINFORCE  (< 1 → Pathwise better)\n'
    '(* = baseline setting)',
    fontsize=14, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'ablation_eps_vs_ratio.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved ablation_eps_vs_ratio.png')


# ─────────────────────────────────────────────────────────────────
# Figure 3: Per-alpha breakdown for each ablation axis
#   4×4 grid: rows = ablation axes, cols = alphas
#   Shows if degradation at small ε depends on learning rate
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 16))

for row, (axis_name, axis_values) in enumerate(ABLATIONS.items()):
    n_vals = len(axis_values)
    pw_colors = cm.Blues(np.linspace(0.4, 0.9, n_vals))
    rf_colors = cm.Reds(np.linspace(0.4, 0.9, n_vals))

    for col, alpha in enumerate(ALPHAS):
        ax = axes[row][col]
        x_pos = np.arange(len(GAPS))

        for vi, val in enumerate(axis_values):
            pw_data = load_ablation('pathwise', axis_name, val)
            rf_data = load_ablation('reinforce', axis_name, val)

            pw_means, pw_std, rf_means, rf_std = [], [], [], []
            for gap in GAPS:
                pw_c = np.array([r['avg_cost'] for r in pw_data[alpha][gap]])
                rf_c = np.array([r['avg_cost'] for r in rf_data[alpha][gap]])
                pw_means.append(np.mean(pw_c))
                pw_std.append(1.96 * np.std(pw_c) / np.sqrt(len(pw_c)))
                rf_means.append(np.mean(rf_c))
                rf_std.append(1.96 * np.std(rf_c) / np.sqrt(len(rf_c)))

            pw_means = np.array(pw_means)
            pw_std = np.array(pw_std)
            rf_means = np.array(rf_means)
            rf_std = np.array(rf_std)

            lbl = AXIS_VAL_FMT[axis_name](val)
            is_baseline = (val == BASELINE[axis_name])
            lw_pw = 2.5 if is_baseline else 1.5

            ax.errorbar(x_pos, pw_means, yerr=pw_std, marker='s', linestyle='-',
                        linewidth=lw_pw, capsize=4, color=pw_colors[vi],
                        label=f'PW {lbl}' if col == 0 else None)
            ax.errorbar(x_pos, rf_means, yerr=rf_std, marker='o', linestyle='--',
                        linewidth=lw_pw * 0.7, capsize=4, color=rf_colors[vi],
                        label=f'RF {lbl}' if col == 0 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(g) for g in GAPS_FLOAT], fontsize=8)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)

        if row == 0:
            ax.set_title(f'α = {alpha}', fontsize=12)
        if col == 0:
            ax.set_ylabel(f'{AXIS_LABELS[axis_name]}\nAvg Holding Cost', fontsize=10)
            ax.legend(fontsize=6, frameon=False, ncol=1)
        if row == 3:
            ax.set_xlabel('Gap size ε', fontsize=10)

fig.suptitle(
    'Per-α Ablation: Pathwise (solid ■) vs REINFORCE (dashed ●)\n'
    r'Rows = ablation axis, Columns = learning rate α  (error bars = 95% CI)',
    fontsize=15, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'ablation_per_alpha.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved ablation_per_alpha.png')
