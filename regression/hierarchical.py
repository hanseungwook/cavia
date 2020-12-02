import higher 
import torch
import torch.nn as nn
from torch.optim import SGD
from dataset import Meta_Dataset, Meta_DataLoader, get_samples  
from misc.rl_utils import get_inner_loss
from misc.status_monitor import StatusMonitor
from misc.meta_memory import MetaMemory


class Hierarchical_Model(nn.Module):
    def __init__(self, base_model, args=None, task=None, logger=None):
        super().__init__()

        self.base_model = base_model 
        self.args = args
        self.task = task
        self.logger = logger
        self.level_max = len(base_model.parameters_all)

        self.status = StatusMonitor(self.level_max)
        self.meta_memory = MetaMemory(args, logger)

    def forward(self, task_batch, level=None, optimizer=SGD, reset=True, is_outer=False):
        """
        Args:
            is_outer (bool): Indicates whether forward is currently at the inner-loop process or
            the outer-loop process. Only needed for TRPO optimization due to adadptation. 
            Default: True.
        """
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
                    self, task.loader['train'], level - 1, self.args, 
                    optimizer=optimizer, reset=reset, 
                    status=self.status, meta_memory=self.meta_memory, is_outer=is_outer)
                test_batch = next(iter(task.loader['test']))
                loss, output = self(test_batch, level=level - 1, is_outer=is_outer)

                # Add to meta-memory
                if not isinstance(output, list) and is_outer is False:
                    self.meta_memory.add(output, key=self.status.key())

                # For next task
                test_loss += loss
                outputs.append(output)

            return test_loss / float(len(task_batch)), outputs


def optimize(model, dataloader, level, args, optimizer, reset, status, meta_memory, is_outer):
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
        for i_task_batch, task_batch in enumerate(dataloader):
            # Update status
            if is_outer and level == 2:
                pass
            else:
                status.update("iteration", level, iteration)

            # Train/optimize up to max_iter # of batches
            if iteration >= args.max_iters[level]:
                return

            loss, outputs = model(task_batch, is_outer=is_outer, level=level)
            if not isinstance(outputs, list) and is_outer is False:
                meta_memory.add(outputs, key=status.key())

            if reset:
                new_param, = optim.step(loss, params=[param_all[level]])
                param_all[level] = new_param
            else:
                if is_outer:
                    return

                optim.zero_grad()
                loss.backward()
                optim.step()

                # For logging
                print("key:", status.key(), iteration)

                before_key = status.get_key_by_id([0, 0, iteration], [0, 0, 0])
                before_memory = meta_memory.get(before_key)
                model.logger.tb_writer.add_scalars("debug/reward0", {"before": before_memory.get_reward()}, iteration)

                after_key = status.get_key_by_id([2, 2, iteration], [0, 0, 0])
                after_memory = meta_memory.get(after_key)
                model.logger.tb_writer.add_scalars("debug/reward0", {"after": after_memory.get_reward()}, iteration)

                before_key = status.get_key_by_id([0, 0, iteration], [0, 1, 0])
                before_memory = meta_memory.get(before_key)
                model.logger.tb_writer.add_scalars("debug/reward1", {"before": before_memory.get_reward()}, iteration)

                after_key = status.get_key_by_id([2, 2, iteration], [0, 1, 0])
                after_memory = meta_memory.get(after_key)
                model.logger.tb_writer.add_scalars("debug/reward1", {"after": after_memory.get_reward()}, iteration)

                # For next outer-loop
                meta_memory.clear()

            iteration += 1   


class Hierarchical_Task(object):
    def __init__(self, task, batch_dict): 
        self.task = task
        self.batch_dict = batch_dict
        self.batch_dict_next = batch_dict[:-1]
        self.level = len(self.batch_dict) - 1
        self.loader = self.get_dataloader_dict()

    def get_dataloader_dict(self):
        return {
            'train': self.get_dataloader(self.batch_dict[-1]['train'], sample_type='train'), 
            'test': self.get_dataloader(self.batch_dict[-1]['test'], sample_type='test')}

    def get_dataloader(self, batch_size, sample_type):
        if self.level == 0:
            return [self.task]  # Format: List of [env_name, task]
        else:
            tasks = get_samples(self.task, batch_size, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=batch_size)


def get_hierarchical_task(batch_dict, args):
    if args.task == "empty":
        task_func_list = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-5x5-v0"]
    elif args.task == "unlock":
        task_func_list = ["MiniGrid-Unlock-Easy-v0", "MiniGrid-Unlock-Easy-v0"]
    elif args.task == "mixture":
        task_func_list = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Unlock-Easy-v0"]
    else:
        raise ValueError("Invalid task option")
    task = Hierarchical_Task(task_func_list, batch_dict)
    return Meta_Dataset(data=[task])
