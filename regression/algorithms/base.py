import torch
import torch.nn.functional as F


class Base(object):
    def __init__(self):
        super(Base, self).__init__()
        self.model = None

    def get_model(self):
        return self.model

    def build_model(self, args, task_family, **kwargs):
        raise NotImplementedError("Subclass has not implemented this method")

    def eval_model(self, inputs, targets, **kwargs):
        raise NotImplementedError("Subclass has not implemented this method")
    
    def eval_model_total(self, **kwargs):
        raise NotImplementedError("Subclass has not implemented this method")

    def update_step(self, optimizer, inputs, targets, **kwargs):
        raise NotImplementedError("Subclass has not implemented this method")

    def meta_backward(self, args, **kwargs):
        raise NotImplementedError("Subclass has not implemented this method")

    def viz_pred(self, inputs, targets, **kwargs):
        raise NotImplementedError("Subclass has not implemented this method")
