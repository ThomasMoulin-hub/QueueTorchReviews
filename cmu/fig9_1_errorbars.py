import json
import numpy as np
import matplotlib.pyplot as plt
import os

CMU_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load and combine data (50 + 950 = 1000 runs) ──
def load_combined(file_base_50, file_base_950):
    with open(os.path.join(CMU_DIR, file_base_50), 'r') as f:
        d1 = json.load(f)
    with open(os.path.join(CMU_DIR, file_base_950), 'r') as f:
        d2 = json.load(f)

    combined = {}
    for alpha in d1:
        combined[alpha] = {}
        for gap in d1[alpha]:
            if gap in d2.get(alpha, {}):
                combined[alpha][gap] = d1[alpha][gap] + d2[alpha][gap]
            else:
                combined[alpha][gap] = d1[alpha][gap]
    return combined


pw = load_combined(
    'pathwise_wc_cmu_multiclass5_all_eps.json',
    'pathwise_wc_cmu_multiclass5_all_eps_950_more_runs.json',
)
rf = load_combined(
    'wc_reinforce_baseline_cmu_B100_multiclass5_all_eps.json',
    'wc_reinforce_baseline_cmu_B100_multiclass5_all_eps_950_more_runs.json',
)

ALPHAS = ['0.01', '0.1', '0.5', '1.0']
GAPS = ['1', '0.5', '0.01']
N_QUEUES = 5

# ─────────────────────────────────────────────────────────────────
# Figure 1: Multi-panel Fig 9.1 — one subplot per ε
#   Bar chart: policy score θ_j per queue, PW vs RF
#   Error bars: ±1 std dev across all runs (pooled over alphas)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(GAPS), figsize=(4.2 * len(GAPS), 5), sharey=True)

for gi, gap in enumerate(GAPS):
    ax = axes[gi]

    # Collect last_iterate across all alphas
    pw_iterates = []
    rf_iterates = []
    for alpha in ALPHAS:
        if gap in pw[alpha]:
            for r in pw[alpha][gap]:
                pw_iterates.append(r['last_iterate'][0])
        if gap in rf[alpha]:
            for r in rf[alpha][gap]:
                rf_iterates.append(r['last_iterate'][0])

    pw_iterates = np.array(pw_iterates)  # (N_runs*4_alphas, 5)
    rf_iterates = np.array(rf_iterates)

    pw_mean = np.mean(pw_iterates, axis=0)
    pw_ci = np.std(pw_iterates, axis=0)
    rf_mean = np.mean(rf_iterates, axis=0)
    rf_ci = np.std(rf_iterates, axis=0)

    x = np.arange(N_QUEUES)
    width = 0.35

    ax.bar(x - width / 2, pw_mean, width, yerr=pw_ci, capsize=4,
           label='Pathwise (B=1)', color='#5B9BD5', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, rf_mean, width, yerr=rf_ci, capsize=4,
           label='REINFORCE (B=100)', color='#ED7D31', edgecolor='black', linewidth=0.5)

    n_pw = len(pw_iterates)
    n_rf = len(rf_iterates)
    ax.set_title(f'ε = {gap}\n(PW: {n_pw} runs, RF: {n_rf} runs)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(j + 1) for j in range(N_QUEUES)])
    ax.set_xlabel('Queue j', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    if gi == 0:
        ax.set_ylabel('Policy Score θ_j', fontsize=12)
        ax.legend(fontsize=8, frameon=False)

fig.suptitle(
    'Fig 9.1 — Learned Policy Scores by Queue (5-class)\n'
    'Error bars = ±1 std dev across 1000 independent runs × 4 learning rates',
    fontsize=13, y=1.03,
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'Fig9_1_errorbars.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved Fig9_1_errorbars.png')


# ─────────────────────────────────────────────────────────────────
# Figure 2: Per-alpha breakdown — one row per α, columns = ε
#   Shows whether variability depends on learning rate
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(ALPHAS), len(GAPS), figsize=(4 * len(GAPS), 3.5 * len(ALPHAS)),
                         sharey='row')

for ai, alpha in enumerate(ALPHAS):
    for gi, gap in enumerate(GAPS):
        ax = axes[ai][gi]

        pw_it, rf_it = [], []
        if gap in pw[alpha]:
            pw_it = np.array([r['last_iterate'][0] for r in pw[alpha][gap]])
        if gap in rf[alpha]:
            rf_it = np.array([r['last_iterate'][0] for r in rf[alpha][gap]])

        x = np.arange(N_QUEUES)
        width = 0.35

        if len(pw_it) > 0:
            pw_mean = np.mean(pw_it, axis=0)
            pw_std = np.std(pw_it, axis=0)
            ax.bar(x - width / 2, pw_mean, width, yerr=pw_std, capsize=3,
                   color='#5B9BD5', edgecolor='black', linewidth=0.4,
                   label='PW' if gi == 0 else None)
        if len(rf_it) > 0:
            rf_mean = np.mean(rf_it, axis=0)
            rf_std = np.std(rf_it, axis=0)
            ax.bar(x + width / 2, rf_mean, width, yerr=rf_std, capsize=3,
                   color='#ED7D31', edgecolor='black', linewidth=0.4,
                   label='RF' if gi == 0 else None)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(j + 1) for j in range(N_QUEUES)], fontsize=8)
        ax.grid(True, axis='y', alpha=0.2)

        if ai == 0:
            n = len(pw_it) if len(pw_it) > 0 else len(rf_it)
            ax.set_title(f'ε = {gap}  ({n} runs)', fontsize=10)
        if gi == 0:
            ax.set_ylabel(f'α = {alpha}\nθ_j', fontsize=10)
            ax.legend(fontsize=7, frameon=False)
        if ai == len(ALPHAS) - 1:
            ax.set_xlabel('Queue j', fontsize=9)

fig.suptitle(
    'Fig 9.1 — Per-α Breakdown (error bars = ±1 std across runs)',
    fontsize=14, y=1.01,
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'Fig9_1_per_alpha.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved Fig9_1_per_alpha.png')


# ─────────────────────────────────────────────────────────────────
# Figure 3: Fig 9.2 style — avg holding cost vs ε
#   Using the combined 1000-run data, per alpha
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

gaps_sorted = sorted(GAPS, key=lambda g: float(g), reverse=False)
x_pos = np.arange(len(gaps_sorted))

for alpha in ALPHAS:
    pw_means, pw_ci = [], []
    rf_means, rf_ci = [], []

    for gap in gaps_sorted:
        if gap in pw[alpha]:
            pw_costs = np.array([r['avg_cost'] for r in pw[alpha][gap]])
            pw_means.append(np.mean(pw_costs))
            pw_ci.append(1.96 * np.std(pw_costs) / np.sqrt(len(pw_costs)))
        else:
            pw_means.append(np.nan)
            pw_ci.append(0)

        if gap in rf[alpha]:
            rf_costs = np.array([r['avg_cost'] for r in rf[alpha][gap]])
            rf_means.append(np.mean(rf_costs))
            rf_ci.append(1.96 * np.std(rf_costs) / np.sqrt(len(rf_costs)))
        else:
            rf_means.append(np.nan)
            rf_ci.append(0)

    ax.errorbar(x_pos, pw_means, yerr=pw_ci, marker='s', linestyle='-',
                linewidth=2.5, capsize=4, label=f'Pathwise (B=1): α={alpha}')
    ax.errorbar(x_pos, rf_means, yerr=rf_ci, marker='o', linestyle='--',
                linewidth=2, capsize=4, alpha=0.85,
                label=f'REINFORCE (B=100): α={alpha}')

ax.set_xticks(x_pos)
ax.set_xticklabels([str(float(g)) for g in gaps_sorted])
ax.invert_xaxis()
ax.set_xlabel('Gap size ε', fontsize=12)
ax.set_ylabel('Holding cost of the avg iterate', fontsize=12)
ax.set_title('Fig 9.2 — 5-class multiserver (1000 runs, 95% CI)', fontsize=13)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'Fig9_2_5class_1000runs.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved Fig9_2_5class_1000runs.png')
