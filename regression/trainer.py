import torch
from misc.rl_utils import get_inner_loss
from misc.utils import log_result

torch.set_num_threads(2)


def adapt(base_model, task, args, logger, meta_ctx):
    ctx = torch.zeros(1, args.n_contexts[0], requires_grad=True, dtype=torch.float32)

    if meta_ctx is None:
        meta_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=False, dtype=torch.float32)

    for _ in range(args.max_iterations[0]):
        loss, _ = get_inner_loss(base_model, task, [ctx, meta_ctx], args)
        grad = torch.autograd.grad(loss, [ctx], create_graph=True)[0]
        ctx = ctx - args.learning_rates[0] * grad

    return ctx


def meta_adapt(base_model, meta_task, args, logger, ctxs):
    assert len(meta_task) == len(ctxs), "Length must be same"
    meta_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=True, dtype=torch.float32)

    for _ in range(args.max_iterations[1]):
        losses = []
        for task, ctx in zip(meta_task, ctxs):
            loss, _ = get_inner_loss(base_model, task, [ctx, meta_ctx], args)
            losses.append(loss)
        loss = sum(losses) / float(len(losses))
        grad = torch.autograd.grad(loss, [meta_ctx], create_graph=True)[0]
        meta_ctx = meta_ctx - args.learning_rates[1] * grad

    return meta_ctx


def train(base_model, hierarchical_task, args, logger):
    for iteration in range(args.max_iterations[-1]):
        # Reset tasks
        hierarchical_task.reset()

        # Log initial reward
        ctx = torch.zeros(1, args.n_contexts[0], requires_grad=False, dtype=torch.float32)
        meta_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=False, dtype=torch.float32)
        for i_meta_task, meta_task in enumerate(hierarchical_task.get_meta_tasks()):
            rewards = []
            for i_task, task in enumerate(meta_task):
                _, memory = get_inner_loss(base_model, task, [ctx, meta_ctx], args)
                rewards.append(memory.get_reward())
            log_result(rewards, iteration, args, logger, prefix=str(i_meta_task) + "/before")

        # Adapt lowest context parameters
        ctxs = []
        for task in hierarchical_task.get_tasks():
            ctx = adapt(base_model, task, args, logger, meta_ctx=None)
            ctxs.append(ctx)

        # Adapt meta-context parameters
        if args.is_hierarchical:
            # Based on adapted context parameters, adapt meta-context parameters
            meta_ctxs = []
            for i_meta_task, meta_task in enumerate(hierarchical_task.get_meta_tasks()):
                ctxs_ = ctxs[i_meta_task * args.batch[1]: (i_meta_task + 1) * args.batch[1]]
                meta_ctx = meta_adapt(base_model, meta_task, args, logger, ctxs=ctxs_)
                meta_ctxs.append(meta_ctx)

            # Based on adapted meta-context parameters, adapt context parameters
            ctxs = []
            for i_meta_task, meta_task in enumerate(hierarchical_task.get_meta_tasks()):
                for i_task, task in enumerate(meta_task):
                    meta_ctx = meta_ctxs[i_meta_task]
                    ctx = adapt(base_model, task, args, logger, meta_ctx=meta_ctx)
                    ctxs.append(ctx)
        else:
            meta_ctxs = [None for _ in hierarchical_task.get_meta_tasks()]

        # Compute test losses
        test_losses = []
        for i_meta_task, meta_task in enumerate(hierarchical_task.get_meta_tasks()):
            rewards = []
            for i_task, task in enumerate(meta_task):
                ctx, meta_ctx = ctxs[i_meta_task * args.batch[1] + i_task], meta_ctxs[i_meta_task]
                test_loss, memory = get_inner_loss(base_model, task, [ctx, meta_ctx], args)
                test_losses.append(test_loss)
                rewards.append(memory.get_reward())
            log_result(rewards, iteration, args, logger, prefix=str(i_meta_task) + "/after")

        # Update base network
        test_loss = sum(test_losses) / float(len(test_losses))
        base_model.optimizer.zero_grad()
        test_loss.backward()
        base_model.optimizer.step()
