import torch
from algorithm.base import Base
import torch.nn.functional as F


class CaviaLevel1(Base):
    def __init__(self, args, log, tb_writer):
        super(CaviaLevel1, self).__init__(args, log, tb_writer)

    def inner_update(self, model, lower_context, higher_context, data):
        input, target = data
        pred = model(input, lower_context, higher_context)
        loss = F.mse_loss(pred, target)

        grad = torch.autograd.grad(loss, lower_context, create_graph=True)[0]
        lower_context = lower_context - grad * self.args.lr_inner

        return lower_context

    def optimize(self, model, higher_context, data):
        lower_context = model.reset_context()
        for _ in range(self.args.n_inner):
            self.inner_update(model, lower_context, higher_context, data)
        return lower_context

    def get_contexts(self, model, higher_context, super_task_data):
        lower_contexts = []
        for data in super_task_data:
            lower_context = self.optimize(model, higher_context, data)
            lower_contexts.append(lower_context)

        assert len(lower_contexts) == len(super_task_data)
        return lower_contexts
