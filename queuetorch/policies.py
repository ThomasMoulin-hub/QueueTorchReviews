import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPriorityPolicy(nn.Module):
    """
    sPR: Policy depends only on a learnable bias (theta), ignoring the state.
    Action probabilities are constant for a given theta.
    """
    def __init__(self, s, q):
        super().__init__()
        self.s = s
        self.q = q
        # Theta is just a bias matrix of size (s, q)
        self.theta = nn.Parameter(torch.randn(s, q))

    def forward(self, queues, time=None):
        batch = queues.size(0)
        # Expand theta to batch size
        logits = self.theta.unsqueeze(0).expand(batch, -1, -1)
        return F.softmax(logits, dim=2)

class SoftMaxWeightPolicy(nn.Module):
    """
    sMW: Policy is a linear function of queue lengths.
    Logits = queues * theta + bias
    """
    def __init__(self, s, q):
        super().__init__()
        self.s = s
        self.q = q
        # Theta maps q inputs to s*q outputs
        self.fc = nn.Linear(q, s * q, bias=True)

    def forward(self, queues, time=None):
        batch = queues.size(0)
        x = self.fc(queues)
        logits = x.reshape(batch, self.s, self.q)
        return F.softmax(logits, dim=2)

class SoftMaxPressurePolicy(nn.Module):
    """
    sMP: In a full implementation, this would use (Q_in - Q_out).
    For this generic implementation, we use a Linear layer similar to sMW 
    but allow for potential future expansion to include downstream info if available.
    For now, it behaves structurally like sMW but is conceptually distinct in experiments.
    """
    def __init__(self, s, q):
        super().__init__()
        self.s = s
        self.q = q
        self.fc = nn.Linear(q, s * q, bias=True)

    def forward(self, queues, time=None):
        batch = queues.size(0)
        x = self.fc(queues)
        logits = x.reshape(batch, self.s, self.q)
        return F.softmax(logits, dim=2)
