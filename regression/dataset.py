import gym
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler


class Meta_Dataset(Dataset):
    def __init__(self, data, target=None):
        self.data = data
        self.target = target
        if target is not None:
            assert len(data) == len(target)
            # self.target = target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.target is None:
            return self.data[idx]
        else: 
            return self.data[idx], self.target[idx]


class Meta_DataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Create indices of batches
        batch_idx = list(BatchSampler(RandomSampler(self.dataset), self.batch_size, drop_last=False))

        # Create dataset of minibatches of the form [mini-batch of (hierarchical tasks), ...]
        mini_dataset = []
        for mini_batch_idx in batch_idx:
            mini_batch = [self.dataset[idx] for idx in mini_batch_idx]
            mini_dataset.append(mini_batch)

        return iter(mini_dataset)
        # return mini_dataset


def get_samples(task, total_batch, sample_type, is_rl=True):
    print("is_rl:", is_rl)

    if isinstance(task, list):
        assert total_batch <= len(task)
        tasks = random.sample(task, total_batch)
    else:
        if is_rl:
            env = gym.make(task)
            lower_tasks = env.sample_tasks(num_tasks=total_batch)
            tasks = [(task, lower_task) for lower_task in lower_tasks]
        else:
            tasks = [task(sample_type) for _ in range(total_batch)]
    return tasks
