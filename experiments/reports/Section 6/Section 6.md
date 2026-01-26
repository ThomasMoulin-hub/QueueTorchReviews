## Section 6 ##
- PPO-WC:
    - Q: Is this equation in section 6 accurate? 
        - work-conserving: assigns a probability of zero to empty queues
            1. if $\epsilon>0$, then when all queue lengths=0, $1{x_l>0} \^ \epsilon =0$, doesn’t prevent division by zero (0/0).
            2. if $\epsilon<0$, if only $x_j = 0$, then numerator = $\epsilon$ $\pi_\theta^{WC}(x)_{ij}\neq0$, defies WC.
               
        ![WC_Softmax Equation](./figs_sec6/WC-Softmax_fig.png)
      
    - Code uses
      $$
      \pi_theta(x)_{ij} = \frac{e^{v_theta (x)_{ij}}\^ x_j}{\sum_{l=1}^n e^{v_theta (x)_{il}}\^ x_l}
      $$

      if not all queues are empty; Otherwise, assign equal weights to each feasible queue.

- Setting: Reentrant_2 config file in codebase. Reentrant-1 (6 classes, 2 servers) in the paper.

- Hyperparameters: 50000 episodes, 100 policy iterations. Actor number as shown in picture. (Used fewer actors in reproduction to save running time, which might cause the results to vary slightly.)

  | Reproduced Fig. 12 | Paper Fig. 12 |
  |--------------------|---------------|
  | <img src="./figs_sec6/reproduced/figure_12.png" width="400"> | <img src="./figs_sec6/paper/figure_12.png" width="400"> |

- Conclusion: The reproduced results generally match those reported in the original paper. PPO-WC > cmu-rule > PPO-BC > PPO Vanilla. 
    - Though plain PPO seems to improve at first, it gets stuck at a bad policy with a high average holding cost and an average queue-length of around 150.
    - Behavior cloning provides a much better initialization, but fails to improve over the cµ-rule. 
    - With the work conserving softmax, even the randomly initialized policy is capable of stabilizing the network – achieving an equivalent cost as the cµ-rule – and is able to outperform the cµ over the course of training.
