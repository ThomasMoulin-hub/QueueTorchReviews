import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# Section 5.2 
## Fig 9.1

with open("/user/xz3355/QueueTorchReviews/cmu/pathwise_wc_cmu_multiclass5.json", "r") as f:
    pw5 = json.load(f)
    
with open("/user/xz3355/QueueTorchReviews/cmu/wc_reinforce_baseline_cmu_multiclass5.json", "r") as f:
    rf5 = json.load(f)
    
alphas = ["0.01", "0.1", "0.5", "1.0"]
gaps = ['0.1'] # [1, 0.5, 0.1, 0.05, 0.01]
n = 5
pw5_results = {}
pw5_count = 0
rf5_results = {}
rf5_count = 0

for i in range(n):
    pw5_results[i] = []
    rf5_results[i] = []
    
for alpha in alphas:
    for gap in gaps:
        for x in pw5[alpha][gap]:
            pw5_count += 1
            for i in range(n):
                pw5_results[i].append(x['last_iterate'][0][i])
        for x in rf5[alpha][gap]:
            rf5_count += 1
            for i in range(n):
                rf5_results[i].append(x['last_iterate'][0][i])

print('pw5:', pw5_results)
print('pw5_count:', pw5_count)
pw5_avg_results = [sum(x)/pw5_count for x in pw5_results.values()]
rf5_avg_results = [sum(x)/rf5_count for x in rf5_results.values()]

print(f"Average Policy Score for PATHWISE: {pw5_avg_results}")
print(f"Average Policy Score for REINFORCE: {rf5_avg_results}")

x = np.arange(n)      
width = 0.35              

plt.bar(x - width/2, pw5_avg_results, width, label='PATHWISE')
plt.bar(x + width/2, rf5_avg_results, width, label='REINFORCE')

plt.xticks(x, ['1', '2', '3', '4', '5'])
plt.xlabel('Queue')
plt.ylabel('Policy Score θ_j')
plt.legend()

plt.tight_layout()
plt.show() 
plt.savefig("/user/xz3355/QueueTorchReviews/cmu/Fig9_1.png")



## Fig9.2
alphas = [0.01, 0.1, 0.5, 1.0]
gaps = [1, 0.5, 0.1, 0.05, 0.01]

# with open("/user/xz3355/queue-learning/cmu/pathwise_wc_cmu_multiclass.json", 'r') as f:
with open("/user/xz3355/QueueTorchReviews/cmu/pathwise_wc_cmu1_multiclass10.json", "r") as f:
    pw10 = json.load(f)
   
# with open("/user/xz3355/queue-learning/cmu/wc_reinforce_baseline_cmu_multiclass.json", "r") as f: 
with open("/user/xz3355/QueueTorchReviews/cmu/wc_reinforce_baseline_cmu_B100_multiclass10.json", "r") as f:
    rf10 = json.load(f)

pw10_results = []
rf10_results = []
pw10_count = 0
rf10_count = 0

for alpha in alphas:
    for gap in gaps:
        pw10_cost_list = []
        rf10_cost_list = []
        
        for x in pw10[str(alpha)][str(gap)]:
            pw10_count += 1
            pw10_cost_list.append(x['avg_cost'])
            
        for x in rf10[str(alpha)][str(gap)]:
            rf10_count += 1
            rf10_cost_list.append(x['avg_cost'])

        pw_n = len(pw10_cost_list)
        rf_n = len(rf10_cost_list)
        pw10_results.append({'alpha':alpha, 'gap': gap, 'avg_cost': np.mean(pw10_cost_list), 'cost_std': 1.96 * np.std(pw10_cost_list) / np.sqrt(pw_n)})
        rf10_results.append({'alpha':alpha, 'gap': gap, 'avg_cost': np.mean(rf10_cost_list), 'cost_std': 1.96 * np.std(rf10_cost_list) / np.sqrt(rf_n)})


fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = list(range(len(gaps))) 
gaps = sorted(gaps, reverse=False)

# ===== 1. Pathwise：每个 alpha 一条线 =====
pw_by_alpha = defaultdict(list)
for r in pw10_results:
    pw_by_alpha[r['alpha']].append(r)

for alpha, records in pw_by_alpha.items():
    records = sorted(records, key=lambda x: x['gap'], reverse=False)

    # gaps  = [r['gap'] for r in records]
    means = [r['avg_cost'] for r in records]
    stds  = [r['cost_std'] for r in records]

    ax.errorbar(
        x, means, yerr=stds,
        marker='s',
        linestyle='-',
        linewidth=2.5,
        capsize=4,
        label=f"Pathwise (B=1): α={alpha}"
    )

# ===== 2. REINFORCE：每个 alpha 一条线 =====
rf_by_alpha = defaultdict(list)
for r in rf10_results:
    rf_by_alpha[r['alpha']].append(r)

for alpha, records in rf_by_alpha.items():
    records = sorted(records, key=lambda x: x['gap'], reverse=False)

    # gaps  = [r['gap'] for r in records]
    means = [r['avg_cost'] for r in records]
    stds  = [r['cost_std'] for r in records]

    ax.errorbar(
        x, means, yerr=stds,
        marker='o',
        linestyle='--',
        linewidth=2,
        capsize=4,
        alpha=0.85,
        label=f"REINFORCE + Value (B=100): α={alpha}"
    )

# ===== 3. 图像样式 =====
ax.set_xticks(x)
ax.set_xticklabels(gaps)
ax.set_xlabel("Gap size ε")
ax.set_ylabel("Holding cost of the avg iterate")
ax.invert_xaxis()
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
plt.savefig("/user/xz3355/QueueTorchReviews/cmu/Fig9_2.png")