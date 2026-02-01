import torch
import torch.nn as nn
from typing import Optional, Callable

class NNHelper:
    """Base class for Koopman operators with neural network utilities"""

    @staticmethod
    def get_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
        """Get learning rate scheduler"""
        name = name.lower()
        if name == 'steplr': return torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))
        elif name == 'exponentiallr': return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.9))
        elif name == 'reducelronplateau': return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=kwargs.get('mode', 'min'), factor=kwargs.get('factor', 0.1), patience=kwargs.get('patience', 10))
        elif name == 'cosineannealinglr': return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50), eta_min=kwargs.get('eta_min', 0))
        elif name == 'cycliclr': return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=kwargs.get('base_lr', 0.001), max_lr=kwargs.get('max_lr', 0.01), step_size_up=kwargs.get('step_size_up', 2000))
        else: raise ValueError(f"Unsupported scheduler: {name}")

    @staticmethod
    def get_optimizer(name: str, parameters, **kwargs) -> torch.optim.Optimizer:
        """Get optimizer"""
        name = name.lower()
        if name == 'adam': return torch.optim.Adam(parameters, lr=kwargs.get('lr', 0.001), weight_decay=kwargs.get('weight_decay', 0.0))
        elif name == 'sgd': return torch.optim.SGD(parameters, lr=kwargs.get('lr', 0.01), momentum=kwargs.get('momentum', 0.9), weight_decay=kwargs.get('weight_decay', 0.0))
        elif name == 'adagrad': return torch.optim.Adagrad(parameters, lr=kwargs.get('lr', 0.01), weight_decay=kwargs.get('weight_decay', 0.0))
        elif name == 'rmsprop': return torch.optim.RMSprop(parameters, lr=kwargs.get('lr', 0.01), weight_decay=kwargs.get('weight_decay', 0.0))
        elif name == 'lbfgs': return torch.optim.LBFGS(parameters, lr=kwargs.get('lr', 0.1), max_iter=kwargs.get('max_iter', 20))
        else: raise ValueError(f"Unsupported optimizer: {name}")

    @staticmethod
    def get_weight_init(name: str, **kwargs) -> Callable:
        """Get weight initializer"""
        name = name.lower()
        if name == 'xavier_uniform': return lambda t: nn.init.xavier_uniform_(t, gain=kwargs.get('gain', 1.0))
        elif name == 'xavier_normal': return lambda t: nn.init.xavier_normal_(t, gain=kwargs.get('gain', 1.0))
        elif name == 'kaiming_uniform': return lambda t: nn.init.kaiming_uniform_(t, mode=kwargs.get('mode', 'fan_in'), nonlinearity=kwargs.get('nonlinearity', 'leaky_relu'))
        elif name == 'kaiming_normal': return lambda t: nn.init.kaiming_normal_(t, mode=kwargs.get('mode', 'fan_in'), nonlinearity=kwargs.get('nonlinearity', 'leaky_relu'))
        elif name == 'normal': return lambda t: nn.init.normal_(t, mean=kwargs.get('mean', 0.0), std=kwargs.get('std', 1.0))
        elif name == 'uniform': return lambda t: nn.init.uniform_(t, a=kwargs.get('a', -0.1), b=kwargs.get('b', 0.1))
        elif name == 'orthogonal': return lambda t: nn.init.orthogonal_(t, gain=kwargs.get('gain', 1.0))
        elif name == 'constant': return lambda t: nn.init.constant_(t, val=kwargs.get('val', 0.0))
        else: return nn.init.xavier_uniform_
    
    @staticmethod
    def get_bias_init(name: str, **kwargs) -> Callable:
        """Get bias initializer"""
        name = name.lower()
        if name == 'zeros': return nn.init.zeros_
        elif name == 'ones': return nn.init.ones_
        elif name == 'normal': return lambda t: nn.init.normal_(t, mean=kwargs.get('mean', 0.0), std=kwargs.get('std', 0.01))
        elif name == 'uniform': return lambda t: nn.init.uniform_(t, a=kwargs.get('a', -0.01), b=kwargs.get('b', 0.01))
        elif name == 'constant': return lambda t: nn.init.constant_(t, val=kwargs.get('val', 0.0))
        else: return nn.init.zeros_
    
    @staticmethod
    def get_activation(name: str, **kwargs) -> Optional[nn.Module]:
        """Get activation function"""
        name = name.lower()
        if name == 'relu': return nn.ReLU(kwargs.get('inplace', False))
        elif name == 'leakyrelu': return nn.LeakyReLU(kwargs.get('negative_slope', 0.01), kwargs.get('inplace', False))
        elif name == 'tanh': return nn.Tanh()
        elif name == 'sigmoid': return nn.Sigmoid()
        elif name == 'elu': return nn.ELU(kwargs.get('alpha', 1.0), kwargs.get('inplace', False))
        elif name in ['swish', 'silu']: return nn.SiLU(kwargs.get('inplace', False))
        elif name == 'gelu': return nn.GELU()
        elif name in ['linear', 'none', '']: return None
        else: raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def init_layer(layer: nn.Module, weight_init: str = 'xavier_uniform', bias_init: str = 'zeros', **kwargs):
        """Initialize layer weights and biases"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            NNHelper.get_weight_init(weight_init, **kwargs)(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            NNHelper.get_bias_init(bias_init, **kwargs)(layer.bias)