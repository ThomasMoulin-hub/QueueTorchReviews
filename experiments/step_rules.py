import torch
import copy


class NormalizedFixed:
    """Current baseline: normalize gradient to unit length, then step with fixed alpha.
    Update: theta -= alpha * (grad / ||grad||)
    The step length is always exactly alpha regardless of gradient magnitude.
    """
    name = 'normalized_fixed'

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, grad, t):
        return self.alpha * grad / torch.linalg.norm(grad)


class NormalizedDiminishing:
    """Normalized gradient with O(1/sqrt(t)) decay.
    Update: theta -= (alpha / sqrt(t+1)) * (grad / ||grad||)
    Classical SGD rate — large early steps for exploration, guaranteed convergence.
    """
    name = 'normalized_diminishing'

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, grad, t):
        return (self.alpha / (t + 1) ** 0.5) * grad / torch.linalg.norm(grad)


class NormalizedPolyak:
    """Normalized gradient with O(1/t) decay.
    Update: theta -= (alpha / (t+1)) * (grad / ||grad||)
    Faster decay than diminishing — stronger convergence guarantee but slower progress.
    """
    name = 'normalized_polyak'

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, grad, t):
        return (self.alpha / (t + 1)) * grad / torch.linalg.norm(grad)


class Adam:
    """Adam optimizer on the raw (unnormalized) gradient.
    Maintains per-coordinate running mean (m) and variance (v) of gradients.
    Update: theta -= alpha * m_hat / (sqrt(v_hat) + eps)
    Adapts step size per coordinate — coordinates with consistent gradient direction
    get larger effective steps; noisy coordinates get smaller steps.
    """
    name = 'adam'

    def __init__(self, alpha, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def step(self, grad, t):
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** (t + 1))
        v_hat = self.v / (1 - self.beta2 ** (t + 1))
        return self.alpha * m_hat / (v_hat.sqrt() + self.eps)


class Adagrad:
    """Adagrad on the raw (unnormalized) gradient.
    Accumulates sum of squared gradients per coordinate.
    Update: theta -= alpha * grad / (sqrt(sum_of_squares) + eps)
    Coordinates that have received large gradients historically get smaller steps.
    Natural annealing without explicit schedule.
    """
    name = 'adagrad'

    def __init__(self, alpha, eps=1e-8):
        self.alpha = alpha
        self.eps = eps
        self.sum_sq = None

    def step(self, grad, t):
        if self.sum_sq is None:
            self.sum_sq = torch.zeros_like(grad)
        self.sum_sq = self.sum_sq + grad ** 2
        return self.alpha * grad / (self.sum_sq.sqrt() + self.eps)


class RMSProp:
    """RMSProp on the raw (unnormalized) gradient.
    Uses exponential moving average of squared gradients, unlike Adagrad
    which accumulates all past squared gradients.
    Update: theta -= alpha * grad / (sqrt(v) + eps)
    where v = beta * v + (1 - beta) * grad^2
    """
    name = 'rmsprop'

    def __init__(self, alpha, beta=0.99, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.v = None

    def step(self, grad, t):
        if self.v is None:
            self.v = torch.zeros_like(grad)
        self.v = self.beta * self.v + (1 - self.beta) * grad ** 2
        return self.alpha * grad / (self.v.sqrt() + self.eps)


class AMSGrad:
    """AMSGrad — Adam variant with guaranteed convergence.
    Maintains max of past bias-corrected second moments to prevent
    the effective learning rate from increasing.
    Update: theta -= alpha * m_hat / (sqrt(v_hat_max) + eps)
    """
    name = 'amsgrad'

    def __init__(self, alpha, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.v_hat_max = None

    def step(self, grad, t):
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
            self.v_hat_max = torch.zeros_like(grad)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** (t + 1))
        v_hat = self.v / (1 - self.beta2 ** (t + 1))
        self.v_hat_max = torch.maximum(self.v_hat_max, v_hat)
        return self.alpha * m_hat / (self.v_hat_max.sqrt() + self.eps)


class UnnormalizedFixed:
    """Raw gradient with fixed step size (no normalization).
    Update: theta -= alpha * grad
    Step length scales with gradient magnitude — larger gradients cause larger steps.
    Requires much smaller alpha than normalized variants.
    """
    name = 'unnormalized_fixed'

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, grad, t):
        return self.alpha * grad


# ── Registry ──

STEP_RULES = {
    'normalized_fixed':       NormalizedFixed,
    # 'normalized_diminishing': NormalizedDiminishing,
    # 'normalized_polyak':      NormalizedPolyak,
    'adam':                    Adam,
    # 'adagrad':                Adagrad,
    'rmsprop':                RMSProp,
    # 'amsgrad':                AMSGrad,
    # 'unnormalized_fixed':     UnnormalizedFixed,
}

# Per-rule alpha ranges: normalized rules can use larger alphas since
# step length is controlled; adaptive/unnormalized need smaller values.
STEP_RULE_ALPHAS = {
    'normalized_fixed':       [0.01, 0.1, 0.5, 1.0],
    'normalized_diminishing': [0.1, 0.5, 1.0, 5.0],
    'normalized_polyak':      [0.1, 0.5, 1.0, 5.0],
    'adam':                    [0.001, 0.01, 0.1, 1.0],
    'adagrad':                [0.01, 0.1, 1.0, 5.0],
    'rmsprop':                [0.001, 0.01, 0.1, 1.0],
    'amsgrad':                [0.001, 0.01, 0.1, 1.0],
    'unnormalized_fixed':     [0.0001, 0.001, 0.01, 0.1],
}


def make_step_rule(name, alpha):
    """Create a fresh step rule instance (needed per worker for stateful rules)."""
    return STEP_RULES[name](alpha)
