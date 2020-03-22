import torch
import random
import numpy as np


class MixutureRegressionTasks(object):
    def __init__(self):

        self.num_inputs = 1
        self.num_outputs = 1
        self.input_range = [-5, 5]
        self.task_types = ["sin", "linear", "quadratic", "cubic"]

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

    def sample_tasks(self, num_tasks):
        target_functions = []

        for i_task in range(num_tasks):
            task_type = random.choice(self.task_types)

            if task_type == "sin":
                amplitude = np.random.uniform(0.1, 5.)
                phase = np.random.uniform(0., np.pi)
                target_function = self.get_sin_function(amplitude, phase)
            elif task_type == "linear":
                slope = np.random.uniform(-3., 3.)
                bias = np.random.uniform(-3., 3.)
                target_function = self.get_linear_function(slope, bias)
            elif task_type == "quadratic":
                slope1 = np.random.uniform(-0.2, 0.2)
                slope2 = np.random.uniform(-2.0, 2.0)
                bias = np.random.uniform(-3., 3.)
                target_function = self.get_quadratic_function(slope1, slope2, bias)
            elif task_type == "cubic":
                slope1 = np.random.uniform(-0.1, 0.1)
                slope2 = np.random.uniform(-0.2, 0.2)
                slope3 = np.random.uniform(-2.0, 2.0)
                bias = np.random.uniform(-3., 3.)
                target_function = self.get_cubic_function(slope1, slope2, slope3, bias)
            else:
                raise ValueError()
            target_functions.append(target_function)

        return target_functions

    def sample_tasks_onehot(self, num_tasks, batch_size):
        q_all = []
        for i in range(num_tasks):
            q = torch.zeros(batch_size, num_tasks)
            q[:, i] = 1.0
            q_all.append(q)
        q_all = torch.cat(q_all)

        return q_all

    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """
        raise NotImplementedError("todo")

        amplitudes = torch.Tensor(np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], batch_size))
        phases = torch.Tensor(np.random.uniform(self.phase_range[0], self.phase_range[1], batch_size))

        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.view(-1)

        outputs = torch.sin(inputs - phases) * amplitudes
        outputs = outputs.unsqueeze(1)

        return torch.stack((inputs, amplitudes, phases)).t(), outputs

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs

    def sample_task(self):
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
        return self.get_target_function(amplitude, phase)
