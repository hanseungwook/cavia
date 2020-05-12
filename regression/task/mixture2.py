##############################################################################
# Parameter and task sampling functions

import numpy as np



def get_sin_params():
    # Sample n_batch number of parameters
    amplitude = np.random.uniform(0.1, 5.)
    phase = np.random.uniform(0., np.pi)
    return amplitude, phase

def get_sin_function(amplitude, phase):
    def sin_function(x):
        # if isinstance(x, torch.Tensor):
        #     return torch.sin(x - phase) * amplitude
        # else:
        return np.sin(x - phase) * amplitude

    return sin_function

def get_linear_params():
    slope = np.random.uniform(-3., 3.)
    bias = np.random.uniform(-3., 3.)

    return slope, bias

def get_linear_function(slope, bias):
    def linear_function(x):
        return slope * x + bias

    return linear_function

# def get_quadratic_params():
#     slope1 = np.random.uniform(-0.2, 0.2)
#     slope2 = np.random.uniform(-2.0, 2.0)
#     bias = np.random.uniform(-3., 3.)

#     return slope1, slope2, bias

# def get_quadratic_function(slope1, slope2, bias):
#     def quadratic_function(x):
#         # if isinstance(x, torch.Tensor):
#         #     return slope1 * torch.pow(x, 2) + slope2 * x + bias
#         # else:
#         return slope1 * np.square(x, 2) + slope2 * x + bias
#         # TypeError: return arrays must be of ArrayType
#     return quadratic_function

# def get_cubic_params():
#     slope1 = np.random.uniform(-0.1, 0.1)
#     slope2 = np.random.uniform(-0.2, 0.2)
#     slope3 = np.random.uniform(-2.0, 2.0)
#     bias = np.random.uniform(-3., 3.)

#     return slope1, slope2, slope3, bias

# def get_cubic_function(slope1, slope2, slope3, bias):
#     def cubic_function(x):
#         # if isinstance(x, torch.Tensor):
#         #     return \
#         #         slope1 * torch.pow(x, 3) + \
#         #         slope2 * torch.pow(x, 2) + \
#         #         slope3 * x + \
#         #         bias
#         # else:
#         return \
#             slope1 * np.power(x, 3) + \
#             slope2 * np.power(x, 2) + \
#             slope3 * x + \
#             bias

#     return cubic_function

task_func_list = [
             (get_sin_params, get_sin_function),
             (get_linear_params, get_linear_function),
            #  (get_quadratic_params, get_quadratic_function),
            #  (get_cubic_params, get_cubic_function),
            ]
