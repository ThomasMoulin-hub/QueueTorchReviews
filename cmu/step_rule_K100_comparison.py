import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

CMU_DIR = os.path.dirname(os.path.abspath(__file__))

GAPS = ['1', '0.5', '0.05', '0.01']
GAPS_FLOAT = [float(g) for g in GAPS]
x_pos = np.arange(len(GAPS))

RULE_LABELS = {
    'normalized_fixed': 'Norm. Fixed',
    'adam':              'Adam',
    'rmsprop':           'RMSProp',
}


def load_json(filename):
    path = os.path.join(CMU_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def get_costs(data, alpha, gaps):
    means, ci = [], []
    for gap in gaps:
        if alpha in data and gap in data[alpha]:
            costs = np.array([r['avg_cost'] for r in data[alpha][gap]])
            means.append(np.mean(costs))
            ci.append(1.96 * np.std(costs) / np.sqrt(len(costs)))
        else:
            means.append(np.nan)
            ci.append(0)
    return np.array(means), np.array(ci)


def best_alpha_costs(data, gaps):
    results = {}
    for gap in gaps:
        best_mean = np.inf
        best = None
        for alpha in data:
            if gap not in data[alpha]:
                continue
            costs = np.array([r['avg_cost'] for r in data[alpha][gap]])
            m = np.mean(costs)
            if m < best_mean:
                best_mean = m
                ci = 1.96 * np.std(costs) / np.sqrt(len(costs))
                best = (m, ci, alpha)
        if best is not None:
            results[gap] = best
    return results


# ── Load data ──
rf_baseline = load_json('wc_reinforce_baseline_cmu_B100_multiclass10.json')
pw_baseline = load_json('pathwise_wc_cmu_multiclass10.json')

k20_pw = {r: load_json(f'pathwise_step_rule_{r}.json') for r in RULE_LABELS}
k100_pw = {r: load_json(f'pathwise_step_rule_{r}_K100.json') for r in RULE_LABELS}
k100_rf = {r: load_json(f'reinforce_step_rule_{r}_K100.json') for r in RULE_LABELS}

print("Loaded all data")


# ─────────────────────────────────────────────────────────────────
# Figure 1: K=20 vs K=100 — best-alpha, Pathwise only
#   Shows improvement from more gradient steps per step rule.
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# RF baseline
rf_best = best_alpha_costs(rf_baseline, GAPS)
ax.errorbar(x_pos, [rf_best[g][0] for g in GAPS], yerr=[rf_best[g][1] for g in GAPS],
            marker='D', linestyle='-', linewidth=3, capsize=5, color='black',
            markersize=8, zorder=10, label='RF (B=100) baseline K=20')

# PW baseline
pw_best = best_alpha_costs(pw_baseline, GAPS)
ax.errorbar(x_pos, [pw_best[g][0] for g in GAPS], yerr=[pw_best[g][1] for g in GAPS],
            marker='s', linestyle='-', linewidth=2.5, capsize=4, color='gray',
            markersize=7, zorder=9, label='PW baseline K=20')

colors_k20 = {'normalized_fixed': '#1f77b4', 'adam': '#ff7f0e', 'rmsprop': '#2ca02c'}
colors_k100 = {'normalized_fixed': '#1f77b4', 'adam': '#ff7f0e', 'rmsprop': '#2ca02c'}

for rule in RULE_LABELS:
    label = RULE_LABELS[rule]
    color = colors_k20[rule]

    # K=20
    if k20_pw[rule]:
        b = best_alpha_costs(k20_pw[rule], GAPS)
        ax.errorbar(x_pos, [b[g][0] for g in GAPS], yerr=[b[g][1] for g in GAPS],
                    marker='o', linestyle=':', linewidth=1.5, capsize=3,
                    color=color, alpha=0.5, label=f'PW {label} K=20')

    # K=100
    if k100_pw[rule]:
        b = best_alpha_costs(k100_pw[rule], GAPS)
        ax.errorbar(x_pos, [b[g][0] for g in GAPS], yerr=[b[g][1] for g in GAPS],
                    marker='s', linestyle='-', linewidth=2.5, capsize=4,
                    color=color, label=f'PW {label} K=100')

ax.set_xticks(x_pos)
ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
ax.invert_xaxis()
ax.set_xlabel('Gap size ε', fontsize=13)
ax.set_ylabel('Holding cost of the avg iterate', fontsize=13)
ax.set_title('K=20 vs K=100: Pathwise Step Rules (best α, 95% CI)', fontsize=14)
ax.legend(frameon=False, fontsize=7.5, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_K20_vs_K100.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_K20_vs_K100.png')


# ─────────────────────────────────────────────────────────────────
# Figure 2: K=100 per-alpha — one subplot per rule
#   PW (solid) vs RF (dashed) at each alpha, K=100 only
# ─────────────────────────────────────────────────────────────────
rules = list(RULE_LABELS.keys())
fig, axes = plt.subplots(1, len(rules), figsize=(6 * len(rules), 5.5), squeeze=False)
axes = axes[0]

for ri, rule in enumerate(rules):
    ax = axes[ri]
    pw_data = k100_pw[rule]
    rf_data = k100_rf[rule]

    if pw_data is None and rf_data is None:
        continue

    all_alphas = sorted(set(
        list(pw_data.keys() if pw_data else []) +
        list(rf_data.keys() if rf_data else [])
    ), key=float)
    n_a = len(all_alphas)
    alpha_colors = cm.viridis(np.linspace(0.15, 0.85, n_a))

    for ai, alpha in enumerate(all_alphas):
        if pw_data and alpha in pw_data:
            m, c = get_costs(pw_data, alpha, GAPS)
            ax.errorbar(x_pos, m, yerr=c, marker='o', linestyle='-',
                        linewidth=2, capsize=4, color=alpha_colors[ai],
                        label=f'PW α={alpha}')
        if rf_data and alpha in rf_data:
            m, c = get_costs(rf_data, alpha, GAPS)
            ax.errorbar(x_pos, m, yerr=c, marker='D', linestyle='--',
                        linewidth=1.5, capsize=3, color=alpha_colors[ai], alpha=0.6,
                        label=f'RF α={alpha}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=11)
    ax.set_ylabel('Avg Holding Cost', fontsize=11)
    ax.set_title(f'{RULE_LABELS[rule]} (K=100)', fontsize=12)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('K=100 Per-Rule: PW (solid) vs RF (dashed) at matching α (95% CI)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_K100_per_rule.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_K100_per_rule.png')


# ─────────────────────────────────────────────────────────────────
# Figure 3: K=100 cost ratio — PW / RF per alpha, one subplot per rule
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(rules), figsize=(6 * len(rules), 5.5), squeeze=False)
axes = axes[0]

for ri, rule in enumerate(rules):
    ax = axes[ri]
    pw_data = k100_pw[rule]
    rf_data = k100_rf[rule]
    if pw_data is None or rf_data is None:
        continue

    common_alphas = sorted(set(pw_data.keys()) & set(rf_data.keys()), key=float)
    n_a = len(common_alphas)
    alpha_colors = cm.viridis(np.linspace(0.15, 0.85, n_a))

    for ai, alpha in enumerate(common_alphas):
        pw_m, _ = get_costs(pw_data, alpha, GAPS)
        rf_m, _ = get_costs(rf_data, alpha, GAPS)
        ratios = pw_m / rf_m
        ax.plot(x_pos, ratios, marker='o', linewidth=2, color=alpha_colors[ai],
                label=f'α={alpha}')

    ax.axhline(1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=11)
    ax.set_ylabel('Cost Ratio (PW / RF)', fontsize=11)
    ax.set_title(f'{RULE_LABELS[rule]} (K=100)', fontsize=12)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)

fig.suptitle('K=100 Cost Ratio: PW / RF at matching α (< 1 → PW better)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_K100_ratio.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_K100_ratio.png')


# ─────────────────────────────────────────────────────────────────
# Figure 4: All K=100 methods — best alpha, PW vs RF side by side
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# Baselines
ax.errorbar(x_pos, [rf_best[g][0] for g in GAPS], yerr=[rf_best[g][1] for g in GAPS],
            marker='D', linestyle='-', linewidth=3, capsize=5, color='black',
            markersize=8, zorder=10, label='RF (B=100) baseline K=20')
ax.errorbar(x_pos, [pw_best[g][0] for g in GAPS], yerr=[pw_best[g][1] for g in GAPS],
            marker='s', linestyle='-', linewidth=2.5, capsize=4, color='gray',
            markersize=7, zorder=9, label='PW baseline K=20')

pw_colors = {'normalized_fixed': '#1f77b4', 'adam': '#ff7f0e', 'rmsprop': '#2ca02c'}
rf_colors = {'normalized_fixed': '#aec7e8', 'adam': '#ffbb78', 'rmsprop': '#98df8a'}

for rule in RULE_LABELS:
    label = RULE_LABELS[rule]
    # PW K=100
    if k100_pw[rule]:
        b = best_alpha_costs(k100_pw[rule], GAPS)
        ax.errorbar(x_pos, [b[g][0] for g in GAPS], yerr=[b[g][1] for g in GAPS],
                    marker='s', linestyle='-', linewidth=2.5, capsize=4,
                    color=pw_colors[rule], label=f'PW {label} K=100')
    # RF K=100
    if k100_rf[rule]:
        b = best_alpha_costs(k100_rf[rule], GAPS)
        ax.errorbar(x_pos, [b[g][0] for g in GAPS], yerr=[b[g][1] for g in GAPS],
                    marker='D', linestyle='--', linewidth=2, capsize=4,
                    color=rf_colors[rule], alpha=0.8, label=f'RF {label} K=100')

ax.set_xticks(x_pos)
ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
ax.invert_xaxis()
ax.set_xlabel('Gap size ε', fontsize=13)
ax.set_ylabel('Holding cost of the avg iterate', fontsize=13)
ax.set_title('K=100: All Methods — PW vs RF (best α, 95% CI)', fontsize=14)
ax.legend(frameon=False, fontsize=7, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_K100_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_K100_all.png')
