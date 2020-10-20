import higher 
import torch
import json
import torch.nn as nn
from torch.optim import SGD
from dataset import Meta_Dataset, Meta_DataLoader, get_samples  
from rl_utils import get_inner_loss
from trpo import trpo_optimization


def get_hierarchical_task(task_func_list, batch_dict):
    task = Hierarchical_Task(task_func_list, batch_dict)
    return Meta_Dataset(data=[task])


class Hierarchical_Memory(object):
    def __init__(self):
        # TODO "Move to trpo"
        self.memories = {}

    def add(self, memory, key):
        self.memories[key] = memory

    def get(self, key):
        return self.memories[key]

    def clear(self):
        self.memories.clear()


hierarchical_memory = Hierarchical_Memory()


##############################################################################
#  Model Hierarchy
class Hierarchical_Model(nn.Module):
    def __init__(self, decoder_model, encoders, args=None, task=None):
        super().__init__()

        self.decoder_model = decoder_model 
        self.encoders = encoders
        self.args = args
        self.task = task

        self.level_max = len(decoder_model.parameters_all)

        self.status_monitor = {}
        self.status_monitor["cur_iter"] = [0 for _ in range(self.level_max)]
        self.status_monitor["i_task"] = [0 for _ in range(self.level_max)]

    def forward(self, task_batch, level=None, optimizer=SGD, reset=True):
        """
        args: minibatch of tasks 
        returns:  mean_test_loss, mean_train_loss, outputs

        Encoder(Adaptation) + Decoder model:
        Takes train_samples, adapt to them, then 
        Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks 
        """
        if level is None:
            level = self.level_max
            
        if level == 0:
            return get_inner_loss(self.decoder_model, task_batch, self.args)
        else:
            test_loss, test_count, outputs = 0, 0, []
            print("task_batch length:", len(task_batch), task_batch)
            for i_task, task in enumerate(task_batch): 
                self.status_monitor["i_task"][level - 1] = i_task
                optimize(
                    self, task.loader['train'], level - 1, self.args, 
                    optimizer=optimizer, reset=reset, status_monitor=self.status_monitor)
                test_batch = next(iter(task.loader['test']))
                loss, output = self(test_batch, level - 1)  # Test only 1 minibatch
                print("[TEST] Adding memory with key: {}\n".format(json.dumps(self.status_monitor)))
                hierarchical_memory.add(output, key=json.dumps(self.status_monitor))

                test_loss += loss
                test_count += 1
                outputs.append(output)

            mean_test_loss = test_loss / test_count

            return mean_test_loss, outputs

    # def adapt(self, task_batch, level=None, optimizer=SGD, reset=True): 
    #     if level is None:
    #         level = self.level_max

    #     if level == 0:
    #         memory = hierarchical_memory.get(key=json.dumps(self.status_monitor))
    #         return get_inner_loss(self.decoder_model, task_batch, self.args, memory=memory)
    #     else:
    #         test_loss, test_count, outputs = 0, 0, []
    #         print("task_batch length:", len(task_batch))

    #         for i_task, task in enumerate(task_batch): 
    #             self.status_monitor["i_task"][level - 1] = i_task
    #             optimize_for_context(
    #                 self, task.loader['train'], level - 1, self.args_dict, 
    #                 optimizer=optimizer, reset=reset, status_monitor=self.status_monitor)
    #             test_batch = next(iter(task.loader['test']))

    #             if level < self.level_max:
    #                 loss, output = self(test_batch, level - 1)  # Test only 1 minibatch
    #             else:
    #                 loss, output = 0., None

    #             test_loss += loss
    #             test_count += 1
    #             outputs.append(output)

    #         mean_test_loss = test_loss / test_count

    #         return mean_test_loss, outputs

    def clear(self):
        self.status_monitor = {}
        self.status_monitor["cur_iter"] = [0 for _ in range(self.level_max)]
        self.status_monitor["i_task"] = [0 for _ in range(self.level_max)]


def optimize(model, dataloader, level, args, optimizer, reset, status_monitor, is_rl=True):
    lr, max_iter = args.lrs[level], args.max_iters[level]
    param_all = model.decoder_model.parameters_all

    # Initialize param & optim
    if reset:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad=True)
        optim = optimizer([param_all[level]], lr=lr)
        optim = higher.get_diff_optim(optim, [param_all[level]])
    else:
        if is_rl:
            pass 
        else:
            optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim

    cur_iter = 0
    while True:
        for i_task_batch, task_batch in enumerate(dataloader):
            status_monitor["cur_iter"][level] = cur_iter

            # Train/optimize up to max_iter # of batches
            if cur_iter >= max_iter:    
                print("Finished optimizing level {}\n".format(level))
                return False

            loss, outputs = model(task_batch, level)
            if type(outputs) is not list:
                print("Adding memory with key: {}".format(json.dumps(status_monitor)))
                hierarchical_memory.add(outputs, key=json.dumps(status_monitor))

            if reset:
                new_param, = optim.step(loss, params=[param_all[level]])   # syntax for diff_optim
                param_all[level] = new_param
            else:
                if is_rl:
                    trpo_optimization(outputs, model)
                    model.clear()
                    hierarchical_memory.clear()
                else:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()  

            cur_iter += 1   


# def optimize_for_context(model, dataloader, level, args_dict, optimizer, reset, status_monitor, is_rl=True):
#     lr, max_iter, logger = get_args(args_dict, level)
#     param_all = model.decoder_model.parameters_all
# 
#     # Initialize param & optim
#     if reset:
#         param_all[level] = torch.zeros_like(param_all[level], requires_grad=True)
#         optim = optimizer([param_all[level]], lr=lr)
#         optim = higher.get_diff_optim(optim, [param_all[level]])
# 
#     cur_iter = 0
#     while True:
#         for i_task_batch, task_batch in enumerate(dataloader):
#             status_monitor["cur_iter"][level] = cur_iter
# 
#             # Train/optimize up to max_iter # of batches
#             if cur_iter >= max_iter:    
#                 print("[ADAPT] Finished optimizing level {}\n".format(level))
#                 return False
# 
#             loss, outputs = model.adapt(task_batch, level)
# 
#             if reset:
#                 new_param, = optim.step(loss, params=[param_all[level]])   # syntax for diff_optim
#                 param_all[level] = new_param
#             else:
#                 return
# 
#             cur_iter += 1   


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

    # total_batch: total # of samples  //  mini_batch: mini batch # of samples
    def get_dataloader(self, batch_size, sample_type):
        if self.level == 0:
            return [self.task]  # Format: List of [env_name, task]
        else:
            tasks = get_samples(self.task, batch_size, sample_type)
            subtask_list = [self.__class__(task, self.batch_dict_next) for task in tasks]
            subtask_dataset = Meta_Dataset(data=subtask_list)
            return Meta_DataLoader(subtask_dataset, batch_size=batch_size)
