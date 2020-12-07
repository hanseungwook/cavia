import gym
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler


class Meta_Dataset(Dataset):
    def __init__(self, data, target=None):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


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


def get_samples(task, batch_size, level):
    if isinstance(task, list):
        tasks = [gym.make(env_name) for env_name in task]
    else:
        lower_tasks = task.sample_tasks(num_tasks=batch_size)
        tasks = [(task, lower_task) for lower_task in lower_tasks]
    return tasks
