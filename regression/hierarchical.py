import higher 
import torch
import torch.nn as nn
from torch.optim import SGD
from dataset import get_samples
from misc.rl_utils import get_inner_loss
from misc.status_monitor import StatusMonitor
from misc.meta_memory import MetaMemory

train_iteration = 0


class Hierarchical_Model(nn.Module):
    def __init__(self, base_model, args=None, logger=None):
        super().__init__()

        self.base_model = base_model 
        self.args = args
        self.logger = logger
        self.level_max = len(base_model.parameters_all)

        self.status = StatusMonitor(self.level_max)
        self.meta_memory = MetaMemory(args, logger)

    def forward(self, task_batch, level=None, optimizer=SGD, reset=True):
        if level is None:
            level = self.level_max
            
        if level == 0:
            return get_inner_loss(self.base_model, task_batch, self.args, self.logger)
        else:
            test_loss, outputs = 0, []
            for i_task, task in enumerate(task_batch):
                self.status.update("i_task", level - 1, i_task)

                # Apply optimization
                optimize(
                    self, task.loader, level - 1, self.args, 
                    optimizer=optimizer, reset=reset, 
                    status=self.status, meta_memory=self.meta_memory)
                if level == 3:
                    return
                test_batch = next(iter(task.loader))
                loss, output = self(test_batch, level=level - 1)

                # Add to meta-memory
                if not isinstance(output, list):
                    self.meta_memory.add(output, key=self.status.key())

                # For next task
                test_loss += loss
                outputs.append(output)

            return test_loss / float(len(task_batch)), outputs


def optimize(model, dataloader, level, args, optimizer, reset, status, meta_memory):
    global train_iteration
    param_all = model.base_model.parameters_all

    if reset:
        # Inner-loop optimizer
        param_all[level] = torch.zeros_like(param_all[level], requires_grad=True)
        optim = optimizer([param_all[level]], lr=args.lrs[level])
        optim = higher.get_diff_optim(optim, [param_all[level]])
    else:
        # Outer-loop optimizer
        optim = optimizer(param_all[level](), lr=0.001)

    iteration = 0
    while True:
        for task_batch in dataloader:
            # Update status
            if reset:
                status.update("iteration", level, iteration)

            # Train/optimize up to max_iter # of batches
            if iteration >= args.max_iters[level] and reset:
                return

            loss, outputs = model(task_batch, level=level)
            if not isinstance(outputs, list):
                meta_memory.add(outputs, key=status.key())

            if reset:
                new_param, = optim.step(loss, params=[param_all[level]])
                param_all[level] = new_param
            else:
                optim.zero_grad()
                loss.backward()
                optim.step()

                # For logging
                before_key = status.get_key_by_id([0, 0, iteration], [0, 0, 0])
                before_memory = meta_memory.get(before_key)
                model.logger.tb_writer.add_scalars("debug/reward0", {"before": before_memory.get_reward()}, train_iteration)

                after_key = status.get_key_by_id([2, 2, iteration], [0, 0, 0])
                after_memory = meta_memory.get(after_key)
                model.logger.tb_writer.add_scalars("debug/reward0", {"after": after_memory.get_reward()}, train_iteration)

                before_key = status.get_key_by_id([0, 0, iteration], [0, 1, 0])
                before_memory = meta_memory.get(before_key)
                model.logger.tb_writer.add_scalars("debug/reward1", {"before": before_memory.get_reward()}, train_iteration)

                after_key = status.get_key_by_id([2, 2, iteration], [0, 1, 0])
                after_memory = meta_memory.get(after_key)
                model.logger.tb_writer.add_scalars("debug/reward1", {"after": after_memory.get_reward()}, train_iteration)

                # For next outer-loop
                meta_memory.clear()
                train_iteration += 1
                return

            iteration += 1   


class Hierarchical_Task(object):
    def __init__(self, task, batch, args, logger): 
        self.task = task
        self.batch = batch
        self.args = args
        self.logger = logger
        self.batch_next = batch[:-1]
        self.level = len(batch) - 1
        self.loader = self.get_dataloader(batch[-1])

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __str__(self):
        if self.level >= 1:
            for task in self.tasks:
                self.logger.log[self.args.log_name].info("Level {}: {}".format(self.level, task))
        if self.level >= 2:
            for loader in self.loader[0]:
                print(loader)
        return ""

    def sample_new_tasks(self, task, batch):
        self.loader = self.get_dataloader(batch[-1], task[0].tasks)
        self.logger.log[self.args.log_name].info("Hierarchical task:")
        print(self)

    def get_dataloader(self, batch_size, tasks=None):
        if self.level == 0:
            self.tasks = [self.task]
            return self.tasks  # Returns: [(env, task)]
        else:
            if tasks is None:
                self.tasks = get_samples(self.task, batch_size, self.level)
            subtask_list = [
                self.__class__(task, self.batch_next, self.args, self.logger) 
                for task in self.tasks]
            return [subtask_list]


def get_hierarchical_task(args, logger):
    if args.task == "empty":
        highest_tasks = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-5x5-v0"]
    elif args.task == "unlock":
        highest_tasks = ["MiniGrid-Unlock-Easy-v0", "MiniGrid-Unlock-Easy-v0"]
    elif args.task == "mixture":
        highest_tasks = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Unlock-Easy-v0"]
    else:
        raise ValueError("Invalid task option")

    hierarchical_task = Hierarchical_Task(highest_tasks, args.batch, args, logger)
    logger.log[args.log_name].info("Hierarchical task:")
    print(hierarchical_task)
    return [hierarchical_task]
