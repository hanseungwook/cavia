import torch
import torch.nn.functional as F
from algorithm.base import Base


class CaviaLevel2(Base):
    def __init__(self, args, log, tb_writer):
        super(CaviaLevel2, self).__init__(args, log, tb_writer)

    def inner_update(self, model, context_models, higher_context, super_task):
        lower_contexts = context_models[0].optimize(model, higher_context, super_task)
        assert len(lower_contexts) == len(super_task)

        # Compute inner-loop loss for higher context based on adapted lower context
        loss = []
        for data, lower_context in zip(super_task, lower_contexts):
            input, target = data
            pred = model(input, lower_context, higher_context)
            loss.append(F.mse_loss(pred, target))
        loss = sum(loss) / float(len(loss))

        grad = torch.autograd.grad(loss, higher_context, create_graph=True)[0]
        higher_context = higher_context - grad * self.args.lr_inner

        return higher_context

    def optimize(self, model, context_models, train_data):
        higher_contexts = []
        for super_task in train_data:
            higher_context = model.reset_context()
            for _ in range(self.args.n_inner):
                higher_context = self.inner_update(model, context_models, higher_context, super_task)
            higher_contexts.append(higher_context)

        assert len(higher_contexts) == len(train_data)
        return higher_contexts
