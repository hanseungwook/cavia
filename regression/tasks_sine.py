import numpy as np
import torch
import IPython

class RegressionTasksSinusoidal:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 1

        self.amplitude_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]

        self.input_range = [-5, 5]

        self.total_num_tasks = 0

        self.tasks_1hot = None
        self.target_functions = None
        self.amplitudes = None
        self.phases = None
        self.q_all = None

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

    @staticmethod
    def get_target_function(amplitude, phase):
        def target_function(x):
            if isinstance(x, torch.Tensor):
                return torch.sin(x - phase) * amplitude
            else:
                return np.sin(x - phase) * amplitude

        return target_function

    def sample_tasks(self, num_tasks, return_specs=False):
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_tasks)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], num_tasks)

        target_functions = []
        for i in range(num_tasks):
            target_functions.append(self.get_target_function(amplitude[i], phase[i]))

        if return_specs:
            return target_functions, amplitude, phase
        else:
            return target_functions

    def sample_tasks_1hot(self, total_num_tasks, batch_num_tasks, batch_size, return_specs=False):
        # Sample whole batch's tasks and create labels
        if not self.target_functions:
            self.total_num_tasks = total_num_tasks
            self.amplitudes = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], total_num_tasks)
            self.phases = np.random.uniform(self.phase_range[0], self.phase_range[1], total_num_tasks)
            self.target_functions = [self.get_target_function(self.amplitudes[i], self.phases[i]) for i in range(total_num_tasks)]
            
            self.q_all = torch.zeros(total_num_tasks, batch_size, total_num_tasks)
            for i in range(total_num_tasks):
                self.q_all[i, :, i] = 1

        # Sample mini-batch's task from whole batch's tasks
        mini_batch_idx = np.random.choice(range(len(self.target_functions)), batch_num_tasks, replace=False)

        if return_specs:
            return np.take(self.target_functions, mini_batch_idx), np.take(self.q_all, mini_batch_idx, axis=0), np.take(self.amplitudes, mini_batch_idx), np.take(self.phases, mini_batch_idx)
        else:
            return np.take(self.target_functions, mini_batch_idx), np.take(self.q_all, mini_batch_idx, axis=0)

    def create_input_range_1hot_labels(self, batch_size, cur_label):
        return cur_label.unsqueeze(0).repeat(batch_size, 1)

    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """

        amplitudes = torch.Tensor(np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], batch_size))
        phases = torch.Tensor(np.random.uniform(self.phase_range[0], self.phase_range[1], batch_size))

        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.view(-1)

        outputs = torch.sin(inputs - phases) * amplitudes
        outputs = outputs.unsqueeze(1)

        return torch.stack((inputs, amplitudes, phases)).t(), outputs
