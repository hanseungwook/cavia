import torch
import numpy as np


class MixutureRegressionTasks(object):
    def __init__(self, args):

        self.args = args
        self.num_inputs = 1
        self.input_range = [-5, 5]
        # self.super_tasks = ["sin", "linear", "quadratic", "cubic"]
        self.super_tasks = ["linear", "quadratic", "cubic"]
        self.reset()

    def reset(self):
        self.params = {}
        self.params["sin"] = [[] for _ in range(2)]
        self.params["linear"] = [[] for _ in range(4)]
        self.params["quadratic"] = [[] for _ in range(4)]
        self.params["cubic"] = [[] for _ in range(4)]

    @staticmethod
    def get_sin_function(amplitude, phase):
        def sin_function(x):
            if isinstance(x, torch.Tensor):
                return torch.sin(x - phase) * amplitude
            else:
                return np.sin(x - phase) * amplitude

        return sin_function

    @staticmethod
    def get_linear_function(slope, bias):
        def linear_function(x):
            return slope * x + bias

        return linear_function

    @staticmethod
    def get_quadratic_function(slope1, slope2, bias):
        def quadratic_function(x):
            if isinstance(x, torch.Tensor):
                return slope1 * torch.pow(x, 2) + slope2 * x + bias
            else:
                return slope1 * np.squre(x, 2) + slope2 * x + bias

        return quadratic_function

    @staticmethod
    def get_cubic_function(slope1, slope2, slope3, bias):
        def cubic_function(x):
            if isinstance(x, torch.Tensor):
                return \
                    slope1 * torch.pow(x, 3) + \
                    slope2 * torch.pow(x, 2) + \
                    slope3 * x + \
                    bias
            else:
                return \
                    slope1 * np.power(x, 3) + \
                    slope2 * np.power(x, 2) + \
                    slope3 * x + \
                    bias

        return cubic_function

    def sample_tasks(self, super_task):
        target_functions = []

        for _ in range(self.args.tasks_per_metaupdate):
            if super_task == "sin":
                amplitude = np.random.uniform(0.1, 5.)
                phase = np.random.uniform(0., np.pi)
                target_function = self.get_sin_function(amplitude, phase)

                self.amplitudes.append(amplitude)
                self.phases.append(phase)
            elif super_task == "linear":
                slope1, slope2 = 0, 0
                slope3 = np.random.uniform(-3., 3.)
                bias = np.random.uniform(-3., 3.)
                target_function = self.get_linear_function(slope3, bias)

                self.params["linear"][0].append(slope1) 
                self.params["linear"][1].append(slope2) 
                self.params["linear"][2].append(slope3)
                self.params["linear"][3].append(bias) 
            elif super_task == "quadratic":
                slope1 = 0
                slope2 = np.random.uniform(-0.2, 0.2)
                slope3 = np.random.uniform(-2.0, 2.0)
                bias = np.random.uniform(-3., 3.)
                target_function = self.get_quadratic_function(slope2, slope3, bias)

                self.params["quadratic"][0].append(slope1) 
                self.params["quadratic"][1].append(slope2) 
                self.params["quadratic"][2].append(slope3)
                self.params["quadratic"][3].append(bias) 
            elif super_task == "cubic":
                slope1 = np.random.uniform(-0.1, 0.1)
                slope2 = np.random.uniform(-0.2, 0.2)
                slope3 = np.random.uniform(-2.0, 2.0)
                bias = np.random.uniform(-3., 3.)
                target_function = self.get_cubic_function(slope1, slope2, slope3, bias)

                self.params["cubic"][0].append(slope1) 
                self.params["cubic"][1].append(slope2) 
                self.params["cubic"][2].append(slope3)
                self.params["cubic"][3].append(bias) 
            else:
                raise ValueError()

            target_functions.append(target_function)

        return target_functions

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs
