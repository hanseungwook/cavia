import torch
import torch.nn.functional as F
from algorithm.base import Base


class CaviaLevel2(Base):
    def __init__(self, args, log, tb_writer):
        super(CaviaLevel2, self).__init__(args, log, tb_writer)

    def inner_update(self, model, context_models, higher_context, super_task_data):
        lower_contexts = context_models[0].get_contexts(model, higher_context, super_task_data)
        assert len(lower_contexts) == len(super_task_data)

        # Compute inner-loop loss for higher context based on adapted lower context
        loss = []
        for data, lower_context in zip(super_task_data, lower_contexts):
            input, target = data
            pred = model(input, lower_context, higher_context)
            loss.append(F.mse_loss(pred, target))

        loss = sum(loss) / float(len(loss))
        grad = torch.autograd.grad(loss, higher_context, create_graph=True)[0]
        higher_context = higher_context - grad * self.args.lr_inner

        return higher_context

    def optimize(self, model, context_models, super_task_data):
        higher_context = model.reset_context()
        for _ in range(self.args.n_inner):
            self.inner_update(model, context_models, higher_context, super_task_data)
        return higher_context

    def get_contexts(self, model, context_models, train_dataset):
        higher_contexts = []
        for super_task_data in train_dataset:
            higher_context = self.optimize(model, context_models, super_task_data)
            higher_contexts.append(higher_context)

        assert len(higher_contexts) == len(train_dataset)
        return higher_contexts

        #     """"""""
        #     # Get pre-adapted lower_context
        #     lower_contexts = []
        #     for i_task in range(args.tasks_per_metaupdate):
        #         task_function = task_functions[i_task]
        #         lower_context = self.inner_update_for_lower_context(
        #             args, model, task_family, task_function, higher_context)
        #         lower_contexts.append(lower_context)
    
        #     # Inner-loop update for higher context
        #     for _ in range(args.n_inner):
        #         higher_inner_losses = []
    
        #         # Compute inner-loop loss for higher context based on adapted lower context
        #         for i_task in range(args.tasks_per_metaupdate):
        #             task_function = task_functions[i_task]
        #             lower_context = lower_contexts[i_task] 
        #             train_inputs = task_family['train'].sample_inputs(args.k_meta_train).to(args.device)
        #             higher_inner_losses.append(
        #                 self.eval_model(model, lower_context, higher_context, train_inputs, task_function))
    
        #         higher_inner_loss = sum(higher_inner_losses) / float(len(higher_inner_losses))
        #         higher_context_grad = torch.autograd.grad(higher_inner_loss, higher_context, create_graph=True)[0]
        #         higher_context = higher_context - higher_context_grad * args.lr_inner
    
        #     log[args.log_name].info("At iteration {} with super task {}, higher-loss: {:.3f}".format(
        #         iteration, super_task, higher_inner_loss.detach().cpu().numpy()))
        #     tb_writer.add_scalars(
        #         "higher_inner_loss", {super_task: higher_inner_loss.detach().cpu().numpy()}, iteration)

        #     """"""""

        #     # Get post-adapted lower_context
        #     lower_contexts = []
        #     for i_task in range(args.tasks_per_metaupdate):
        #         task_function = task_functions[i_task]
        #         lower_context = self.inner_update_for_lower_context(
        #             args, model, task_family, task_function, higher_context)
        #         lower_contexts.append(lower_context)
    
        #     # Get meta-loss for the base model
        #     for i_task in range(args.tasks_per_metaupdate):
        #         task_function = task_functions[i_task]
        #         lower_context = lower_contexts[i_task] 
        #         test_inputs = task_family['test'].sample_inputs(args.k_meta_train).to(args.device)
        #         meta_losses.append(self.eval_model(
        #             model, lower_context, higher_context, test_inputs, task_function))

        #     import sys
        #     sys.exit()
    
        #     # Visualize prediction
        #     if iteration % 10 == 0:
        #         self.vis_prediction(
        #             model, lower_context, higher_context, test_inputs, task_function, super_task, iteration, args)

        # return sum(meta_losses) / float(len(meta_losses))
