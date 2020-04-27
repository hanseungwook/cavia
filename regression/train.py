import torch.optim as optim
import torch.nn.functional as F


def get_task_family(args):
    if args.task == 'mixture':
        from task import mixture
        task_family = mixture.MixutureRegressionTasks(args)
    else:
        raise NotImplementedError()

    return task_family


def get_model(args):
    n_arch = args.architecture
    n_context = args.n_context_params

    if args.model_type == "CAVIA":
        from model.cavia import Cavia
        MODEL = Cavia
        n_arch[0] += n_context * args.n_context_models
    else:
        raise ValueError()
        
    model = MODEL(
        n_arch=n_arch, 
        n_context=n_context, 
        device=args.device).to(args.device)

    return model


def get_data(task_family, args):
    train_data, val_data = [], []

    for super_task in task_family.super_tasks:
        train_super_task, val_super_task = [], []
        task_functions = task_family.sample_tasks(super_task)

        for task_function in task_functions:
            train_input = task_family.sample_inputs(args.k_meta_train).to(args.device)
            train_target = task_function(train_input)
            train_super_task.append((train_input, train_target))

            val_input = task_family.sample_inputs(args.k_meta_train).to(args.device)
            val_target = task_function(val_input)
            val_super_task.append((val_input, val_target))

        train_data.append(train_super_task)
        val_data.append(val_super_task)

    return train_data, val_data


def run(args, log, tb_writer):
    # Set tasks
    task_family = get_task_family(args)

    # Set model that includes theta params
    model = get_model(args)
    meta_optimizer = optim.Adam(model.parameters(), args.lr_meta)

    # Set context models
    from algorithm.cavia_level1 import CaviaLevel1
    context_models = [CaviaLevel1(args, log, tb_writer)]

    if args.n_context_models == 2:
        from algorithm.cavia_level2 import CaviaLevel2
        context_models.append(CaviaLevel2(args, log, tb_writer))

    assert len(context_models) == args.n_context_models

    # Begin meta-train
    for iteration in range(2000):
        # Sample train and validation data
        train_data, val_data = get_data(task_family, args)

        # Get higher contexts
        # Returns higher contexts for all super tasks
        higher_contexts = context_models[-1].optimize(model, context_models, train_data)

        # Get lower contexts
        # Returns lower contexts for all super tasks and sub-tasks per super task
        lower_contexts = []
        for super_task, higher_context in zip(train_data, higher_contexts):
            lower_contexts.append(context_models[0].optimize(model, higher_context, super_task))

        # Get meta_loss
        meta_loss = []
        for i_super_task, super_task in enumerate(val_data):
            higher_context = higher_contexts[i_super_task]

            for i_sub_task, sub_task in enumerate(super_task):
                lower_context = lower_contexts[i_super_task][i_sub_task]
                input, target = sub_task
                pred = model(input, lower_context, higher_context)
                meta_loss.append(F.mse_loss(pred, target))
        meta_loss = sum(meta_loss) / float(len(meta_loss))

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        log[args.log_name].info("At iteration {}, meta-loss: {:.3f}".format(
            iteration, meta_loss.detach().cpu().numpy()))
        tb_writer.add_scalar("Meta loss:", meta_loss.detach().cpu().numpy(), iteration)

        # # Visualize result
        # vis_pca(higher_contexts, task_family, iteration, args)
