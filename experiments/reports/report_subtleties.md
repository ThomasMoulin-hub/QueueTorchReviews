# Report on Reproducibility Subtleties

This report outlines discrepancies and subtleties encountered when attempting to reproduce the results from the paper. `Every Figure is reproducible through the reproducing_figs notebook inside the notebooks folder`.
## Section 5.1 and 5.3
### 1. Implementation of Policies

The paper provides formulas for `softPriority`, `softMaxWeight`, and `softMaxPressure`. However, the repositories (`queue-learning` and `QueueTorch`) contain a `policy.py` file where these policies are implemented as `nn.Linear` modules from PyTorch, which seems inconsistent with the paper's description.

Upon inspection, the docstring for `sMP` (softMaxPressure) in `policy.py` states:
> "For now, it behaves structurally like sMW but is conceptually distinct in experiments."

However, running experiments yields different results compared to those obtained using the original code found in `gradient_comparison_maxpressure.py`.

**Key Observation:** Whether running Experiment 5.1 with the original code or using the policies provided in `policy.py`, the obtained results differ significantly from those reported in the paper.

### 2. Cosine Similarity Measure (5.1)

The paper mentions a cosine similarity measure, but this metric does not appear to be implemented in the provided codebase.

Attempts were made to reproduce the figures by:
1. Using Ethan's code to generate the data.
2. Manually computing the cosine similarity.

The results obtained from this process do not match the values indicated in the paper. Furthermore, the specific code used to generate the figures in the paper could not be located.

### 3. Naming Conventions and Configuration Mismatches

There are discrepancies between the naming conventions used in the paper and the codebase, particularly regarding the "Reentrant" scenarios.

*   **Paper "Reentrant-2"**: Corresponds to `re-reentrant` in the codebase.
*   **Paper "Reentrant"**: Corresponds to `reentrant` in the codebase.

Additionally, the number of classes differs by a factor of 3 when mapping paper descriptions to filenames:
*   Example: `Reentrant-2` with 9 classes in the paper corresponds to `re-reentrant_3` in the codebase.


### 4. Comparative Report: DDES Compliance of admission_control (the one I made) vs. buffer_control (the one I suppose was used in part 5.3 leading to figure 11)
The analysis concludes that admission_control.py aligns with the theoretical requirements of the DDES paper, whereas buffer_control.py maybe contains errors in simulation state management. The latter breaks the mathematical assumptions required for calculating valid pathwise gradients.
1. Violation of the Augmented State ($s_k$).
According to Section 3.2 of the DDES paper, the system state is Markovian only if augmented: $s_k=(x_k,z_k)$, where $x_k$ is the queue length and $z_k$ includes auxiliary data (residual inter-arrival times $\tau_{Ak}$ and residual workloads $w_k$).

   - `admission_control.py` (Valid): This script correctly preserves the full state. It maintains the continuity of simulation by carrying over the auxiliary data ($z_k$) between updates via `init_obs`. It respects the joint distribution of the system state.
   - `buffer_control.py` (Invalid): This script suffers from "temporal amnesia." By calling a hard `reset()` that only preserves queue lengths (`init_queues`), it discards the auxiliary timing data ($z_k$) and forces the generation of new random times. This invalidates the Markovian property of the DDES framework.

 
2. Integrity of Pathwise Gradients (The Counterfactual).
The core of the DDES method (Section 4) relies on infinitesimal counterfactual analysis: calculating how the objective function changes with respect to control parameters under a fixed realization of exogenous noise ($xi_1:N$).
   - `admission_control.py` (Valid): It uses a "clipping" mechanism (min$(x_k,L)$) to adjust buffer states without resetting the timeline. This ensures that the sequence of random events (arrivals/service times) remains identical across gradient steps, satisfying the "Reparameterization Trick" requirements.
   - `buffer_control.py` (Invalid): Because it regenerates random times at every step via `reset()`, it effectively creates a "jump" in the trajectory. This breaks the correlation required for differentiation, turning the gradient estimator into a high-variance random walk rather than a precise pathwise derivative.


## Section 5.2 and section 6 
1. Section 5.2
   - Figure 9 (Left): Stated REINFORCE(B=1000) in the plot, but in the description below stated comparison between PATHWISE(B=1) and REINFORCE(B=100). It's also REINFORCE(B=100) in the paragraph.
   - Figure 9 (Left): Not clear on whether it's an aggregated result (average) of multiple alphas, or one specific alpha.
   - Figure 9 (Right): Are error bars std error/ std deviation or Confidence Interval?
  
   - $\beta$ used in the softmin operation of PATHWISE for this section, is not clearly specified in paper.
      - Appendix A stated an inverse temperature of $\beta = 10$ (I assume might be for Section 7?), but would yield bad results (i.e., not able to learn the cmu rule).
      - Codebase uses a temperature of 1e-6. This would lead to reasonable results where PATHWISE is able to learn the cmu rule with only 1 trajectory.

   - Resultwise, PATHWISE(B=1) and REINFORCE(B=100) seem to have similar performance, unlike in the paper where PATHWISE(B=1) significantly outperforms REINFORCE(B=100).
  
3. Section 6
- PPO-WC:
    - Q: Is this equation in section 6 accurate?
      
      <img src="Section 6/figs_sec6/WC-Softmax_fig.png" width="500" center> 
      
        - work-conserving: assigns a probability of zero to empty queues
            1. if $\epsilon>0$, then when all queue lengths=0, $1 \{ x_l>0 \} \wedge \epsilon =0$, doesn’t prevent division by zero (0/0).
            2. if $\epsilon<0$, if only $x_j = 0$, then numerator $\pi_\theta^{WC}(x)_{ij} \wedge \epsilon \neq0$, defies WC.        
      
    - Code uses
      <img width="180" height="50" alt="image" src="https://github.com/user-attachments/assets/0b38f4fe-3530-49ef-9412-3030aa3348db" />
      if not all queues are empty; Otherwise, assign equal weights to each feasible queue.

