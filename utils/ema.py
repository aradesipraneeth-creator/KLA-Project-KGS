"""
Exponential Moving Average (EMA) for model weights to improve generalization and stability.
"""

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Maintains a moving average of model parameters.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """
        Updates EMA parameters: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())

        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

        # Also copy non-parameter buffers (e.g. running stats if any)
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, buf in model_buffers.items():
            if name in ema_buffers:
                ema_buffers[name].data.copy_(buf.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        self.ema_model = self.ema_model.to(device)
        return self
