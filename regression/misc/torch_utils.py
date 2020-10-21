from torch.nn.utils.convert_parameters import _check_param_device
from torch.distributions import Categorical


def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    else:
        raise NotImplementedError()
    return distribution


def vector_to_parameters(vector, parameters):
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer + num_param].view_as(param).data)
        pointer += num_param
