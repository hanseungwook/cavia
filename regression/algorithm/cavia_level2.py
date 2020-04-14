import os
import torch
import matplotlib.pyplot as plt
from algorithm.base import Base


class CaviaLevel2(Base):
    def __init__(self):
        super(CaviaLevel2, self).__init__()

    def get_meta_loss(self, model, task_family, args, log, tb_writer, iteration):
        meta_losses = []
    
        for super_task in task_family["train"].super_tasks:
            task_functions = task_family["train"].sample_tasks(super_task)
            higher_context = model.reset_context()
    
            # Get adapted lower_context
            lower_contexts = []
            for i_task in range(args.tasks_per_metaupdate):
                task_function = task_functions[i_task]
                lower_context = self.inner_update_for_lower_context(
                    args, model, task_family, task_function, higher_context)
                lower_contexts.append(lower_context)
    
            # Inner-loop update for higher context
            for _ in range(args.n_inner):
                higher_inner_losses = []
    
                # Compute inner-loop loss for higher context based on adapted lower context
                for i_task in range(args.tasks_per_metaupdate):
                    task_function = task_functions[i_task]
                    lower_context = lower_contexts[i_task] 
                    train_inputs = task_family['train'].sample_inputs(args.k_meta_train).to(args.device)
                    higher_inner_losses.append(
                        self.eval_model(model, lower_context, higher_context, train_inputs, task_function))
    
                higher_inner_loss = sum(higher_inner_losses) / float(len(higher_inner_losses))
                higher_context_grad = torch.autograd.grad(higher_inner_loss, higher_context, create_graph=True)[0]
                higher_context = higher_context - higher_context_grad * args.lr_inner
    
            log[args.log_name].info("At iteration {} with super task {}, higher-loss: {:.3f}".format(
                iteration, super_task, higher_inner_loss.detach().cpu().numpy()))
            tb_writer.add_scalars(
                "higher_inner_loss", {super_task: higher_inner_loss.detach().cpu().numpy()}, iteration)
    
            # Get meta-loss for the base model
            for i_task in range(args.tasks_per_metaupdate):
                task_function = task_functions[i_task]
                lower_context = lower_contexts[i_task] 
                test_inputs = task_family['test'].sample_inputs(args.k_meta_train).to(args.device)
                meta_losses.append(self.eval_model(
                    model, lower_context, higher_context, test_inputs, task_function))
    
            # Visualize prediction
            if iteration % 10 == 0:
                self.vis_prediction(
                    model, lower_context, higher_context, test_inputs, task_function, super_task, iteration, args)
    
        return sum(meta_losses) / float(len(meta_losses))

    def vis_prediction(self, model, lower_context, higher_context, inputs, task_function, super_task, iteration, args):
        # Create directories
        if not os.path.exists("./logs/n_inner" + str(args.n_inner)):
            os.makedirs("./logs/n_inner" + str(args.n_inner))
    
        outputs = model(inputs, lower_context, higher_context).detach().cpu().numpy()
        targets = task_function(inputs).detach().cpu().numpy()
    
        plt.figure()
        plt.scatter(inputs, outputs, label="pred")
        plt.scatter(inputs, targets, label="gt")
        plt.legend()
        plt.title(super_task + "_iteration" + str(iteration))
    
        plt.savefig("logs/n_inner" + str(args.n_inner) + "/iteration" + str(iteration).zfill(3) + "_" + super_task + ".png")
        plt.close()  
