### Comparative Analysis of Scripts `admission_control.py` and `buffer_control.py`

Although both scripts share a common base for simulating queueing networks, there are fundamental differences in their calculations and configurations. These differences explain the divergent results observed, particularly on complex networks such as re-entrant systems.

#### 1. Differences in Service Policy (MaxWeight)
This is the most critical difference for re-entrant networks (multi-server):
* **`admission_control.py`**: Calculates `logits` by multiplying priorities (`mu * h`) by the `network` matrix **before** taking the argmax. Therefore, each server chooses the best queue among those it is physically capable of serving.
* **`buffer_control.py`**: Takes a global `argmax` across all queues in the system for every server. If the globally chosen queue is not served by a specific server (according to `network`), that server ends up "idle" for that step or ends up serving all its queues with uniform probability (line 65). This makes the policy extremely inefficient in re-entrant networks where servers have distinct roles.

#### 2. Simulation State Management (Reset vs. Warm-start)
* **`admission_control.py`**: Calls `dq.reset()` at every optimization iteration. The simulation starts from empty queues (or the initial state) every time. With a short `T` (100), the system may not reach its steady state.
* **`buffer_control.py`**: Reuses the final state of the queues from the previous iteration as the initial state for the next one (`init_queues = queues.detach()`). This allows for a continuous simulation that better reflects the long-term impact of parameter changes.

#### 3. Configuration and Optimization Parameters
* **YAML Files**: The physical configurations (`h`, `mu`, `network`, `lam`) are identical in the respective folders, but the `reentrant` and `re-reentrant` files themselves differ:
    * `reentrant_2`: Arrivals at queues 0 and 2; the job exits after queue 4.
    * `re-reentrant_2`: Arrivals only at queue 0; the job is re-injected into queue 2 after queue 4 (additional re-entry loop).

#### 4. Stability and Precision
* `admission_control.py` includes a function `patch_env_for_stability` that forces zero arrival rates to a tiny value (`1e-20`) to prevent numerical instabilities; this is not present in `buffer_control.py`.

### Conclusion
The scripts do not perform the same calculations because the server decision logic (policy) and the simulation continuity differ. The base configurations are the same, but the very structure of the `reentrant` and `re-reentrant` environments (defined in the associated `.npy` files) imposes constraints that only `admission_control.py` handles correctly via its server-localized MaxWeight policy.
