import torch
import torch.nn.functional as F


class Base(object):
    def __init__(self):
        super(Base, self).__init__()

    def eval_model(self, model, lower_context, higher_context, inputs, target_fnc):
        outputs = model(inputs, lower_context, higher_context)
        targets = target_fnc(inputs)
        return F.mse_loss(outputs, targets)

    def inner_update_for_lower_context(self, args, model, task_family, task_function, higher_context=None):
        lower_context = model.reset_context()
        train_inputs = task_family['train'].sample_inputs(args.k_meta_train).to(args.device)
    
        for _ in range(args.n_inner):
            lower_inner_loss = self.eval_model(model, lower_context, higher_context, train_inputs, task_function)
            lower_context_grad = torch.autograd.grad(lower_inner_loss, lower_context, create_graph=True)[0]
            lower_context = lower_context - lower_context_grad * args.lr_inner
         
        return lower_context
