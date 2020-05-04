import torch
from algorithm.base import Base
import torch.nn.functional as F


class CaviaLevel1(Base):
    def __init__(self, args, log, tb_writer):
        super(CaviaLevel1, self).__init__(args, log, tb_writer)

    def inner_update(self, model, lower_context, higher_context, task):
        input, target = task
        pred = model(input, lower_context, higher_context)
        loss = F.mse_loss(pred, target)

        grad = torch.autograd.grad(loss, lower_context, create_graph=True)[0]
        lower_context = lower_context - grad * self.args.lr_inner

        return lower_context

    def optimize(self, model, higher_context, super_task):
        lower_contexts = []
        for task in super_task:
            lower_context = model.reset_context()
            for _ in range(self.args.n_inner):
                lower_context = self.inner_update(model, lower_context, higher_context, task)
            lower_contexts.append(lower_context)

        assert len(lower_contexts) == len(super_task)
        return lower_contexts
