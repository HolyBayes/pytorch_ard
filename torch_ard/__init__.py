_author__ = 'Artem Ryzhikov'
__version__ = '0.2.4'
__all__ = ['LinearARD', 'Conv2dARD', 'get_ard_reg', 'get_dropped_params_ratio', 'ELBOLoss']

from .torch_ard import LinearARD, Conv2dARD, get_ard_reg, get_dropped_params_ratio, ELBOLoss
