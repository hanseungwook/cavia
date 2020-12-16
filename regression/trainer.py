import torch
from misc.rl_utils import get_inner_loss


def adapt(base_model, task, args, logger, ctx=None, super_ctx=None, level=0):
    if ctx is None:
        ctx = torch.zeros(1, args.n_contexts[0], requires_grad=True)

    if super_ctx is None:
        super_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=True)

    for _ in range(args.max_iters[level]):
        loss = get_inner_loss(base_model, task, [ctx, super_ctx], args, logger)
        if level == 0:
            grad = torch.autograd.grad(loss, [ctx], create_graph=True)[0]
            ctx = ctx - args.lrs[level] * grad
        else:
            raise NotImplementedError()

    logger.log[args.log_name].info("Finished optimizing at level {}: {}, {}".format(level, ctx, super_ctx))
    if level == 0:
        return ctx
    else:
        return super_ctx


def train(base_model, hierarchical_task, args, logger):
    # Initialize context parameters
    while True:
        # Reset tasks
        hierarchical_task.reset()

        # First, adapt 0-level context
        ctxs = []
        for task in hierarchical_task.get_tasks():
            ctx = adapt(base_model, task, args, logger, super_ctx=None, level=0)
            ctxs.append(ctx)

        print("yo")
        import sys
        sys.exit()

        # Based on adapted 0-level context, adapt 1-level context variables
        for super_task in hierarchical_task.get_super_tasks():
            super_ctx = adapt(base_model, super_task, ctx=ctx)

        # Based on adapted 1-level context, adapt 0-level context variables again
        for task in hierarchical_task.get_tasks():
            ctx = adapt(base_model, task, super_ctx=super_ctx)
