##############################################################################
# Parameter and task sampling functions

import numpy as np
import torch
from utils import batch_wrapper


input_range_1d = [-5, 5]

def x_coord0_params(batch, grid=False):         # Full inputs over the whole regression input range
    if batch == 101 or grid:
        return torch.linspace(input_range_1d[0], input_range_1d[1], steps=batch).unsqueeze(1)
    else:
        return torch.rand(batch, 1) * (input_range_1d[1] - input_range_1d[0]) + input_range_1d[0]

###############################

def sine1_fnc(params):
    amplitude, phase = params
    def lv0_fnc(x):
        return np.sin(x - phase) * amplitude
    return lv0_fnc, x_coord0_params

def sine1_params():       # Sample n_batch number of parameters
    amplitude = np.random.uniform(0.1, 5.)
    phase = np.random.uniform(0., np.pi)
    return torch.tensor([amplitude, phase])

def sine_gen():
    return (sine1_fnc, batch_wrapper(sine1_params))  # sine1_task

###############################

def line1_fnc(params):
    slope, bias = params
    def lv0_fnc(x):
        return slope * x + bias
    return lv0_fnc, x_coord0_params

def line1_params():
    slope = np.random.uniform(-3., 3.)
    bias = np.random.uniform(-3., 3.)
    return torch.tensor([slope, bias])

def line_gen():
    return (line1_fnc, batch_wrapper(line1_params))  #line1_task

###############################

def sine_linear2_fnc(param2):
    a1, a0 = param2
    def lv1_fnc(params1):
        b1, b0 = params1
        def lv0_fnc(x):
            return (a1*x+a0) * np.sin(b1*x+b0)
        return lv0_fnc, x_coord0_params
    return lv1_fnc, batch_wrapper(sine1_params)

def sine_linear_gen():
    return  (sine_linear2_fnc, batch_wrapper(line1_params))  # sine_linear2_task

###############################


# def get_quadratic_function():

#     def get_quadratic_params():
#         slope1 = np.random.uniform(-0.2, 0.2)
#         slope2 = np.random.uniform(-2.0, 2.0)
#         bias = np.random.uniform(-3., 3.)
#         return slope1, slope2, bias    
    
#     slope1, slope2, bias = get_quadratic_params()
    
#     def quadratic_function(x):
#         if isinstance(x, torch.Tensor):
#             return slope1 * torch.pow(x, 2) + slope2 * x + bias
#         else:
#             return slope1 * np.squre(x, 2) + slope2 * x + bias

#     return quadratic_function


# def get_cubic_function():
    
#     def get_cubic_params():
#         slope1 = np.random.uniform(-0.1, 0.1)
#         slope2 = np.random.uniform(-0.2, 0.2)
#         slope3 = np.random.uniform(-2.0, 2.0)
#         bias = np.random.uniform(-3., 3.)
#         return slope1, slope2, slope3, bias
    
#     slope1, slope2, slope3, bias = get_cubic_params()
    
#     def cubic_function(x):
#         if isinstance(x, torch.Tensor):
#             return \
#                 slope1 * torch.pow(x, 3) + \
#                 slope2 * torch.pow(x, 2) + \
#                 slope3 * x + \
#                 bias
#         else:
#             return \
#                 slope1 * np.power(x, 3) + \
#                 slope2 * np.power(x, 2) + \
#                 slope3 * x + \
#                 bias

#     return cubic_function





########################

# def get_quadratic_params():
#     slope1 = np.random.uniform(-0.2, 0.2)
#     slope2 = np.random.uniform(-2.0, 2.0)
#     bias = np.random.uniform(-3., 3.)

#     return slope1, slope2, bias

# def get_quadratic_function(slope1, slope2, bias):
#     def quadratic_function(x):
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
#         return \
#             slope1 * np.power(x, 3) + \
#             slope2 * np.power(x, 2) + \
#             slope3 * x + \
#             bias

#     return cubic_function


####################################


# from torchmeta.utils.data import Task, MetaDataset

# class Sinusoid(MetaDataset):
#     """
#     Simple regression task, based on sinusoids, as introduced in [1].

#     Parameters
#     ----------
#     num_samples_per_task : int
#         Number of examples per task.

#     num_tasks : int (default: 1,000,000)
#         Overall number of tasks to sample.

#     noise_std : float, optional
#         Amount of noise to include in the targets for each task. If `None`, then
#         nos noise is included, and the target is a sine function of the input.

#     transform : callable, optional
#         A function/transform that takes a numpy array of size (1,) and returns a
#         transformed version of the input.

#     target_transform : callable, optional
#         A function/transform that takes a numpy array of size (1,) and returns a
#         transformed version of the target.

#     dataset_transform : callable, optional
#         A function/transform that takes a dataset (ie. a task), and returns a 
#         transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

#     Notes
#     -----
#     The tasks are created randomly as random sinusoid function. The amplitude
#     varies within [0.1, 5.0], the phase within [0, pi], and the inputs are
#     sampled uniformly in [-5.0, 5.0]. Due to the way PyTorch handles datasets,
#     the number of tasks to be sampled needs to be fixed ahead of time (with
#     `num_tasks`). This will typically be equal to `meta_batch_size * num_batches`.
#     """

#     def __init__(self, num_samples_per_task, num_tasks=1000000,
#                  noise_std=None, transform=None, target_transform=None, dataset_transform=None):
#         super().__init__(meta_split='train', target_transform=target_transform, dataset_transform=dataset_transform)
#         self.num_samples_per_task = num_samples_per_task
#         self.num_tasks = num_tasks
#         self.noise_std = noise_std
#         self.transform = transform

#         self._input_range = np.array([-5.0, 5.0])
#         self._amplitude_range = np.array([0.1, 5.0])
#         self._phase_range = np.array([0, np.pi])

#         self._amplitudes = None
#         self._phases = None

#     @property
#     def amplitudes(self):
#         if self._amplitudes is None:
#             self._amplitudes = self.np_random.uniform(self._amplitude_range[0], self._amplitude_range[1], size=self.num_tasks)
#         return self._amplitudes

#     @property
#     def phases(self):
#         if self._phases is None:
#             self._phases = self.np_random.uniform(self._phase_range[0], self._phase_range[1], size=self.num_tasks)
#         return self._phases

#     def __len__(self):
#         return self.num_tasks

#     def __getitem__(self, index):
#         amplitude, phase = self.amplitudes[index], self.phases[index]
#         task = SinusoidTask(index, amplitude, phase, self._input_range, self.noise_std, self.num_samples_per_task, self.transform, self.target_transform, np_random=self.np_random)

#         if self.dataset_transform is not None:
#             task = self.dataset_transform(task)

#         return task


# class SinusoidTask(Task):
#     def __init__(self, index, amplitude, phase, input_range, noise_std,
#                  num_samples, transform=None, target_transform=None,
#                  np_random=None):
#         super(SinusoidTask, self).__init__(index, None) # Regression task
#         self.amplitude = amplitude
#         self.phase = phase
#         self.input_range = input_range
#         self.num_samples = num_samples
#         self.noise_std = noise_std

#         self.transform = transform
#         self.target_transform = target_transform

#         if np_random is None:
#             np_random = np.random.RandomState(None)

#         self._inputs = np_random.uniform(input_range[0], input_range[1],
#             size=(num_samples, 1))
#         self._targets = amplitude * np.sin(self._inputs - phase)
#         if (noise_std is not None) and (noise_std > 0.):
#             self._targets += noise_std * np_random.randn(num_samples, 1)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, index):
#         input, target = self._inputs[index], self._targets[index]

#         if self.transform is not None:
#             input = self.transform(input)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return (input, target)
