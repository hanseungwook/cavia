import os
import matplotlib.pyplot as plt
from algorithm.base import Base


class CaviaLevel1(Base):
    def __init__(self):
        super(CaviaLevel1, self).__init__()

    def get_meta_loss(self, model, task_family, args, log, tb_writer, iteration):
        meta_losses = []
    
        for super_task in task_family["train"].super_tasks:
            higher_context = None
            task_functions = task_family["train"].sample_tasks(super_task)
    
            for i_task in range(args.tasks_per_metaupdate):
                # Get adapted lower_context
                task_function = task_functions[i_task]
                lower_context = self.inner_update_for_lower_context(
                    args, model, task_family, task_function, higher_context)

                # Get meta loss
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
        if not os.path.exists("./logs/1_level_n_inner" + str(args.n_inner)):
            os.makedirs("./logs/1_level_n_inner" + str(args.n_inner))
    
        outputs = model(inputs, lower_context, higher_context).detach().cpu().numpy()
        targets = task_function(inputs).detach().cpu().numpy()
    
        plt.figure()
        plt.scatter(inputs, outputs, label="pred")
        plt.scatter(inputs, targets, label="gt")
        plt.legend()
        plt.title(super_task + "_iteration" + str(iteration))
    
        plt.savefig(
            "logs/1_level_n_inner" + str(args.n_inner) + "/iteration" + 
            str(iteration).zfill(3) + "_" + super_task + ".png")
        plt.close()  
