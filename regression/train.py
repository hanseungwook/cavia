import os
import torch.optim as optim
import matplotlib.pyplot as plt


def get_task_family(args):
    task_family = {}

    if args.task == 'sine':
        from task import sine
        task_family['train'] = sine.RegressionTasksSinusoidal()
        task_family['valid'] = sine.RegressionTasksSinusoidal()
        task_family['test'] = sine.RegressionTasksSinusoidal()
    elif args.task == 'mixture':
        from task import mixture
        task_family['train'] = mixture.MixutureRegressionTasks(args)
        task_family['valid'] = mixture.MixutureRegressionTasks(args)
        task_family['test'] = mixture.MixutureRegressionTasks(args)
    else:
        raise NotImplementedError()

    return task_family


def get_model_decoder(args):
    n_arch = args.architecture
    n_context = args.n_context_params

    if args.model_type == "CAVIA":
        from model.cavia import Cavia
        MODEL = Cavia
        n_arch[0] += n_context * args.n_context_models
    else:
        raise ValueError()
        
    model_decoder = MODEL(
        n_arch=n_arch, 
        n_context=n_context, 
        device=args.device).to(args.device)

    return model_decoder


def run(args, log, tb_writer):
    # Set tasks
    task_family = get_task_family(args)

    # Set model
    model = get_model_decoder(args)
    meta_optimizer = optim.Adam(model.parameters(), args.lr_meta)

    # Set algorithm
    if args.n_context_models == 1:
        from algorithm.cavia_level1 import CaviaLevel1
        algorithm = CaviaLevel1()
    elif args.n_context_models == 2:
        from algorithm.cavia_level2 import CaviaLevel2
        algorithm = CaviaLevel2()
    else:
        raise ValueError()

    # Begin meta-train
    for iteration in range(2000):
        meta_optimizer.zero_grad()
        meta_loss = algorithm.get_meta_loss(model, task_family, args, log, tb_writer, iteration) 
        meta_loss.backward()
        meta_optimizer.step()

        log[args.log_name].info("At iteration {}, meta-loss: {:.3f}".format(
            iteration, meta_loss.detach().cpu().numpy()))
        tb_writer.add_scalar("Meta loss:", meta_loss.detach().cpu().numpy(), iteration)


def vis_prediction(model, lower_context, higher_context, inputs, task_function, super_task, iteration, args):
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
