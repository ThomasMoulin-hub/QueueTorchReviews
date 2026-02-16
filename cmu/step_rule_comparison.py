import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

CMU_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data sources ──
# Baselines (10-class): structure {alpha: {gap: [runs]}}
PATHWISE_BASELINE_FILE = 'pathwise_wc_cmu_multiclass10.json'
REINFORCE_B100_FILE = 'wc_reinforce_baseline_cmu_B100_multiclass10.json'

# Step rule files (pathwise only): same {alpha: {gap: [runs]}} structure
STEP_RULE_FILES = {
    'normalized_fixed':       'pathwise_step_rule_normalized_fixed.json',
    'normalized_diminishing': 'pathwise_step_rule_normalized_diminishing.json',
    'normalized_polyak':      'pathwise_step_rule_normalized_polyak.json',
    'adam':                    'pathwise_step_rule_adam.json',
    'adagrad':                'pathwise_step_rule_adagrad.json',
    'rmsprop':                'pathwise_step_rule_rmsprop.json',
    'amsgrad':                'pathwise_step_rule_amsgrad.json',
    'unnormalized_fixed':     'pathwise_step_rule_unnormalized_fixed.json',
}

# Nice display names
STEP_RULE_LABELS = {
    'normalized_fixed':       'Norm. Fixed',
    'normalized_diminishing': 'Norm. Diminishing',
    'normalized_polyak':      'Norm. Polyak',
    'adam':                    'Adam',
    'adagrad':                'Adagrad',
    'rmsprop':                'RMSProp',
    'amsgrad':                'AMSGrad',
    'unnormalized_fixed':     'Unnorm. Fixed',
}

# Gaps shared across step rule files and baselines
GAPS = ['1', '0.5', '0.05', '0.01']
GAPS_FLOAT = [float(g) for g in GAPS]


def load_json(filename):
    path = os.path.join(CMU_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def get_costs(data, alpha, gaps):
    """Return (means, ci95) arrays for a given alpha across gaps."""
    means, ci = [], []
    for gap in gaps:
        if alpha in data and gap in data[alpha]:
            costs = np.array([r['avg_cost'] for r in data[alpha][gap]])
            means.append(np.mean(costs))
            ci.append(1.96 * np.std(costs) / np.sqrt(len(costs)))
        else:
            means.append(np.nan)
            ci.append(0)
    return means, ci


# ── Load all data ──
pw_baseline = load_json(PATHWISE_BASELINE_FILE)
rf_baseline = load_json(REINFORCE_B100_FILE)

step_rule_data = {}
for name, fname in STEP_RULE_FILES.items():
    d = load_json(fname)
    if d is not None:
        step_rule_data[name] = d

available_rules = list(step_rule_data.keys())
rf_alphas = sorted(rf_baseline.keys(), key=float) if rf_baseline else []

print(f"Loaded {len(available_rules)} step rules: {available_rules}")
print(f"RF(B100) alphas: {rf_alphas}")


# ─────────────────────────────────────────────────────────────────
# Figure 1: Per-alpha Fig 9.2 — one subplot per α
#   Each subplot: RF(B=100) at that α vs all PW step rules at that α.
#   X-axis: gap ε (decreasing), Y-axis: avg holding cost
# ─────────────────────────────────────────────────────────────────
n_alphas = len(rf_alphas)
fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas, 5.5), squeeze=False)
axes = axes[0]

rule_colors = cm.tab10(np.linspace(0, 1, max(len(available_rules), 1)))
x_pos = np.arange(len(GAPS))

for ai, alpha in enumerate(rf_alphas):
    ax = axes[ai]

    # REINFORCE B=100 at this alpha
    rf_means, rf_ci = get_costs(rf_baseline, alpha, GAPS)
    ax.errorbar(x_pos, rf_means, yerr=rf_ci, marker='D', linestyle='-',
                linewidth=3, capsize=5, color='black', markersize=8, zorder=10,
                label='REINFORCE (B=100)')

    # PW baseline at this alpha
    if pw_baseline and alpha in pw_baseline:
        pw_means, pw_ci = get_costs(pw_baseline, alpha, GAPS)
        ax.errorbar(x_pos, pw_means, yerr=pw_ci, marker='s', linestyle='-',
                    linewidth=2.5, capsize=4, color='gray', markersize=7, zorder=9,
                    label='PW baseline')

    # All step rules that have this alpha
    for ri, rule_name in enumerate(available_rules):
        rule = step_rule_data[rule_name]
        if alpha not in rule:
            continue
        means, ci = get_costs(rule, alpha, GAPS)
        label = STEP_RULE_LABELS.get(rule_name, rule_name)
        ax.errorbar(x_pos, means, yerr=ci, marker='o', linestyle='--',
                    linewidth=2, capsize=4, color=rule_colors[ri], alpha=0.85,
                    label=f'PW: {label}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=12)
    ax.set_title(f'α = {alpha}', fontsize=13)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    if ai == 0:
        ax.set_ylabel('Holding cost of the avg iterate', fontsize=12)

fig.suptitle(
    'Per-α: REINFORCE (B=100) vs Pathwise Step Rules — 10-class (95% CI)',
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_per_alpha.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_per_alpha.png')


# ─────────────────────────────────────────────────────────────────
# Figure 2: Per-alpha cost ratio (PW step rule / RF) — one subplot per α
#   Horizontal line at 1.0 = parity.
#   < 1 means PW step rule beats REINFORCE at the same α.
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas, 5.5), squeeze=False)
axes = axes[0]

for ai, alpha in enumerate(rf_alphas):
    ax = axes[ai]

    rf_means, _ = get_costs(rf_baseline, alpha, GAPS)
    rf_means = np.array(rf_means)

    # PW baseline ratio
    if pw_baseline and alpha in pw_baseline:
        pw_means, _ = get_costs(pw_baseline, alpha, GAPS)
        ratios = np.array(pw_means) / rf_means
        ax.plot(x_pos, ratios, marker='s', linewidth=2.5, color='gray',
                markersize=7, zorder=9, label='PW baseline')

    # Step rule ratios
    for ri, rule_name in enumerate(available_rules):
        rule = step_rule_data[rule_name]
        if alpha not in rule:
            continue
        means, _ = get_costs(rule, alpha, GAPS)
        ratios = np.array(means) / rf_means
        label = STEP_RULE_LABELS.get(rule_name, rule_name)
        ax.plot(x_pos, ratios, marker='o', linewidth=2, color=rule_colors[ri],
                alpha=0.85, label=f'PW: {label}')

    ax.axhline(1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=12)
    ax.set_title(f'α = {alpha}', fontsize=13)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    if ai == 0:
        ax.set_ylabel('Cost Ratio (PW / RF)', fontsize=12)

fig.suptitle(
    'Per-α Cost Ratio: Pathwise Step Rules / REINFORCE (B=100) — 10-class\n'
    '< 1 → Pathwise better',
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_ratio.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_ratio.png')


# ─────────────────────────────────────────────────────────────────
# Figure 3: One subplot per step rule — all its alphas vs RF at matching alphas
#   Gives full picture of each optimizer across its entire alpha range.
# ─────────────────────────────────────────────────────────────────
n_rules_total = len(available_rules)
ncols = min(3, n_rules_total)
nrows = (n_rules_total + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

for ri, rule_name in enumerate(available_rules):
    row, col = divmod(ri, ncols)
    ax = axes[row][col]

    rule = step_rule_data[rule_name]
    rule_alphas = sorted(rule.keys(), key=float)
    n_a = len(rule_alphas)
    alpha_colors = cm.viridis(np.linspace(0.15, 0.85, n_a))

    for ai, alpha in enumerate(rule_alphas):
        # PW step rule line (solid)
        pw_means, pw_ci = get_costs(rule, alpha, GAPS)
        ax.errorbar(x_pos, pw_means, yerr=pw_ci, marker='o', linestyle='-',
                    linewidth=2, capsize=4, color=alpha_colors[ai],
                    label=f'PW α={alpha}')

        # RF at matching alpha (dashed, same color)
        if rf_baseline and alpha in rf_baseline:
            rf_means, rf_ci = get_costs(rf_baseline, alpha, GAPS)
            ax.errorbar(x_pos, rf_means, yerr=rf_ci, marker='D', linestyle='--',
                        linewidth=1.5, capsize=3, color=alpha_colors[ai], alpha=0.6,
                        label=f'RF α={alpha}')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(g) for g in GAPS_FLOAT])
    ax.invert_xaxis()
    ax.set_xlabel('Gap size ε', fontsize=11)
    ax.set_ylabel('Avg Holding Cost', fontsize=11)
    label = STEP_RULE_LABELS.get(rule_name, rule_name)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for ri in range(n_rules_total, nrows * ncols):
    row, col = divmod(ri, ncols)
    axes[row][col].set_visible(False)

fig.suptitle(
    'Per-Rule Breakdown — PW (solid) vs RF (dashed) at matching α\n'
    '10-class, 95% CI',
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(CMU_DIR, 'step_rule_per_rule.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved step_rule_per_rule.png')
