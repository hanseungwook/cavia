import torch
import numpy as np


class ReplayMemory():
    def __init__(self):
        self.obs = []
        self.logprob = []
        self.reward = []
        self.done = []

    def _get_mask(self, done):
        done = torch.stack(done, dim=1)
        traj_batch_size, ep_max_timestep = done.shape[0], done.shape[1]
        done = torch.cat((torch.zeros(traj_batch_size, 1), done), dim=1)
        done = done[:, :int(ep_max_timestep)]
        mask = 1. - done
        return mask

    def add(self, obs, logprob, reward, done):
        self.obs.append(obs)
        self.logprob.append(logprob)
        self.reward.append(torch.from_numpy(np.array(reward)))
        self.done.append(torch.FloatTensor(done.astype(float)))

    def sample(self):
        return self.obs, self.logprob, self.reward, self._get_mask(self.done)
