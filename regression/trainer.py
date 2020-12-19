import torch
from misc.rl_utils import get_inner_loss


def adapt(base_model, task, args, logger, meta_ctx=None):
    ctx = torch.zeros(1, args.n_contexts[0], requires_grad=True)

    if meta_ctx is None:
        meta_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=False)

    for _ in range(args.max_iters[0]):
        loss, _ = get_inner_loss(base_model, task, [ctx, meta_ctx], args, logger)
        grad = torch.autograd.grad(loss, [ctx], create_graph=True)[0]
        ctx = ctx - args.lrs[0] * grad

    return ctx


def meta_adapt(base_model, meta_task, args, logger, ctxs):
    assert len(meta_task) == len(ctxs), "Length must be same"
    meta_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=True)

    for _ in range(args.max_iters[1]):
        losses = []
        for task, ctx in zip(meta_task, ctxs):
            loss, _ = get_inner_loss(base_model, task, [ctx, meta_ctx], args, logger)
            losses.append(loss)
        loss = sum(losses) / float(len(losses))
        grad = torch.autograd.grad(loss, [meta_ctx], create_graph=True)[0]
        meta_ctx = meta_ctx - args.lrs[1] * grad

    return meta_ctx


def train(base_model, hierarchical_task, args, logger):
    # Initialize context parameters
    for train_iteration in range(int(1e6)):
        # Reset tasks
        hierarchical_task.reset()

        # Log initial reward
        rewards = []
        ctx = torch.zeros(1, args.n_contexts[0], requires_grad=False)
        meta_ctx = torch.zeros(1, args.n_contexts[1], requires_grad=False)
        for task in hierarchical_task.get_tasks():
            _, memory = get_inner_loss(base_model, task, [ctx, meta_ctx], args, logger)
            rewards.append(memory.get_reward())
        reward = sum(rewards) / float(len(rewards))
        logger.tb_writer.add_scalars("reward", {"before": reward}, train_iteration)

        # First, adapt 0-level context
        ctxs = []
        for task in hierarchical_task.get_tasks():
            ctx = adapt(base_model, task, args, logger, meta_ctx=None)
            ctxs.append(ctx)

        # Based on adapted 0-level context, adapt 1-level context variables
        meta_ctxs = []
        for i_meta_task, meta_task in enumerate(hierarchical_task.get_meta_tasks()):
            ctxs_ = ctxs[i_meta_task * args.batch[1]: (i_meta_task + 1) * args.batch[1]]
            meta_ctx = meta_adapt(base_model, meta_task, args, logger, ctxs=ctxs_)
            meta_ctxs.append(meta_ctx)

        # Based on adapted 1-level context, adapt 0-level context variables again
        ctxs = []
        for i_task, task in enumerate(hierarchical_task.get_tasks()):
            meta_ctx = meta_ctxs[0] if i_task < args.batch[1] else meta_ctxs[1]
            ctx = adapt(base_model, task, args, logger, meta_ctx=meta_ctx)
            ctxs.append(ctx)

        # Compute test loss and update base network
        rewards, test_losses = [], []
        for i_task, task in enumerate(hierarchical_task.get_tasks()):
            ctx = ctxs[i_task]
            meta_ctx = meta_ctxs[0] if i_task < args.batch[1] else meta_ctxs[1]
            test_loss, memory = get_inner_loss(base_model, task, [ctx, meta_ctx], args, logger)
            test_losses.append(test_loss)
            rewards.append(memory.get_reward())

        reward = sum(rewards) / float(len(rewards))
        logger.tb_writer.add_scalars("reward", {"after": reward}, train_iteration)

        test_loss = sum(test_losses) / float(len(test_losses))
        base_model.optimizer.zero_grad()
        test_loss.backward()
        base_model.optimizer.step()
