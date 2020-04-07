import time
from copy import deepcopy
import numpy as np


class Logger:

    def __init__(self, best_model):
        self.train_loss = []
        self.train_conf = []

        self.valid_loss = [0]
        self.valid_conf = [0]

        self.test_loss = [0]
        self.test_conf = [0]

        # self.best_valid_model = None
        self.update_best_model(best_model)

    def update_best_model(self, model):
        self.best_valid_model = deepcopy(model)

    def print_info(self, iter_idx, start_time):
        print(
            'Iter {:<4} - time: {:<5} - [train] loss: {:<6} (+/-{:<6}) - [valid] loss: {:<6} (+/-{:<6}) - [test] loss: {:<6} (+/-{:<6})'.format(
                iter_idx,
                int(time.time() - start_time),
                np.round(self.train_loss[-1], 4),
                np.round(self.train_conf[-1], 4),
                np.round(self.valid_loss[-1], 4),
                np.round(self.valid_conf[-1], 4),
                np.round(self.test_loss[-1], 4),
                np.round(self.test_conf[-1], 4),
            )
        )
