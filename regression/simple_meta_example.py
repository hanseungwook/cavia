# TODO Multiple tasks
import gym
import higher
import torch
import torch.nn as nn
import numpy as np
from linear_baseline import LinearFeatureBaseline, get_return
from torch.distributions import Categorical
from torch.optim import Adam, SGD
from gym_minigrid.wrappers import VectorObsWrapper
from replay_memory import ReplayMemory
from multiprocessing_env import SubprocVecEnv
from tensorboardX import SummaryWriter
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence


traj_batch_size = 5
ep_max_timestep = 20

tb_writer = SummaryWriter('./logs/tb_{0}'.format("rl_merge"))


def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    else:
        raise NotImplementedError()
    return distribution


def make_env(env_name):
    def _make_env():
        env = gym.make(env_name)
        env.max_steps = min(env.max_steps, ep_max_timestep)  # TODO Set horizon from config file
        env.reset_task(task=(4, 4))  # TODO Eliminate same task
        return VectorObsWrapper(env)        
    return _make_env


max_iter_list = [3, 2, 500]  # Level 0, level 1, level 2
n_ctx = [3, 3]
n_input = 6 + sum(n_ctx)  # TODO Remove hard-coding
lr = 0.001
layers = nn.Linear(n_input, 64), nn.Linear(64, 64), nn.Linear(64, 7)  # TODO Remove hard-coding
data = torch.randn(10, 1), torch.randn(10, 1)
env = SubprocVecEnv([make_env("MiniGrid-Unlock-Easy-v0") for _ in range(traj_batch_size)])
linear_baseline = LinearFeatureBaseline(6)  # TODO Remove hard-coding
counter = 0


class Hierarchical_Memory(object):
    def __init__(self):
        # self.memories = {}
        self.memories = []

    def summary(self):
        # LEVEL 0: Level0 iter(Level1 iter + 1)
        for level in self.memories.keys():
            print("At level {}, size is: {}".format(level, len(self.memories[level])))

    # def add(self, level, memory):
    #     if level not in self.memories.keys():
    #         self.memories[level] = []
    #     self.memories[level].append(memory)

    def add(self, memory):
        self.memories.append(memory)

    def clear(self):
        self.memories.clear()


hierarchical_memory = Hierarchical_Memory()


def collect_trajectory(base_model):
    global counter

    # Initialize memory
    memory = ReplayMemory()

    obs = env.reset()
    score = 0.

    for timestep in range(ep_max_timestep):
        # Get action and logprob
        categorical = base_model(obs)
        action = categorical.sample()
        logprob = categorical.log_prob(action)

        # Take step in the environment
        action = action.cpu().numpy().astype(int)
        next_obs, reward, done, _ = env.step(action)

        # Add to memory
        memory.add(
            obs=obs, 
            action=torch.from_numpy(action), 
            logprob=logprob, 
            reward=reward,
            done=done)

        # For logging
        score += np.mean(reward)

        # For next timestep
        obs = next_obs

    print("score:", score)
    counter += 1
    tb_writer.add_scalar("score", score, counter)

    return memory


def get_inner_loss(base_model, task, pointer=None):
    if pointer is None:
        memory = collect_trajectory(base_model)
        obs, action, logprob, reward, mask = memory.sample()
        logprob = torch.stack(logprob, dim=1)
    else:
        print("pointer:", pointer)
        memory = hierarchical_memory.memories[pointer]
        obs, action, logprob, reward, mask = memory.sample()
        categorical = base_model(torch.from_numpy(np.stack(obs, axis=1)).float())
        action = torch.from_numpy(np.stack(action, axis=1)).float()
        logprob = categorical.log_prob(action)  # NOTE Replace logprob

    # Get baseline
    value = linear_baseline(obs, reward, mask)

    # Get REINFORCE loss with baseline
    logprob = logprob * mask
    return_ = get_return(reward, mask)
    loss = torch.mean(torch.sum(logprob * (return_ - value), dim=1))

    return -loss, memory


def make_ctx(n):
    return torch.zeros(1, n, requires_grad=True)


class BaseModel(nn.Module):
    def __init__(self, n_ctx, *layers):
        super().__init__()

        self.layers = nn.ModuleList(list(layers))        
        self.parameters_all = [make_ctx(n) for n in n_ctx] + [self.layers.parameters]
        self.nonlin = nn.ReLU()

    def forward(self, x, layers=None, ctx_=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if ctx_ is None:
            ctx = self.parameters_all[:-1]
        else:
            ctx = ctx_

        if len(x.shape) == 3:
            ctx = torch.cat(ctx, dim=1)
            x = torch.cat((x, ctx.expand(x.shape[0], x.shape[1], -1)), dim=-1)
        else:
            ctx = torch.cat(ctx, dim=1)
            x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)

        layers_ = self.layers if layers is None else layers

        for i, layer in enumerate(layers_):
            x = layer(x)
            x = self.nonlin(x) if i < len(layers_) - 1 else x

        return Categorical(logits=x)

    def reset_context(self):
        print("[INFO] reset context ...")
        self.parameters_all = [make_ctx(n) for n in n_ctx] + [self.layers.parameters]


class Hierarchical_Model(nn.Module):
    def __init__(self, submodel): 
        super().__init__()

        self.submodel = submodel 
        self.level_max = len(submodel.parameters_all)
        self.pointer = -1

    def forward(self, data, level=None, optimizer=None, reset=True): 
        if level is None:
            level = self.level_max

        if level == 0:
            return get_inner_loss(self.submodel, data)
        else:
            optimize(
                model=self, 
                data=data, 
                level=level - 1, 
                max_iter=max_iter_list[level - 1], 
                optimizer=optimizer, 
                reset=reset)
            test_loss, memory = self(data, level - 1)

        print("Return test_loss ...", test_loss, level)
        # print(hierarchical_memory.summary())

        return test_loss, memory

    def adapt(self, data, level=None, optimizer=None, reset=True): 
        if level is None:
            level = self.level_max

        if level == 0:
            print("pointer:", self.pointer)
            self.pointer += 1
            return get_inner_loss(self.submodel, data, self.pointer)
        else:
            optimize_for_context(
                model=self, 
                data=data, 
                level=level - 1, 
                max_iter=max_iter_list[level - 1], 
                optimizer=optimizer, 
                reset=reset)

            if level < self.level_max:
                test_loss, memory = self.adapt(data, level - 1)
            else:
                test_loss, memory = None, None

        print("[ADAPT] Return test_loss ...", test_loss, level)

        return test_loss, memory


def surrogate_loss(memory, model, old_pi=None):
    # TODO RESET CONTEXT
    print("======================GETTING NEW CONTEXT")
    model.submodel.reset_context()
    model.adapt(data, reset=False)
    ctx = model.submodel.parameters_all[:-1]
    model.pointer = -1  # Reset pointer back to origin

    with torch.set_grad_enabled(old_pi is None):
        obs, action, logprob, reward, mask = memory.sample()
        pi = model.submodel(torch.from_numpy(np.stack(obs, axis=1)).float(), ctx_=ctx)

        if old_pi is None:
            old_pi = detach_distribution(pi)

        action = torch.stack(action, dim=1)
        log_ratio = (pi.log_prob(action) - old_pi.log_prob(action))
        ratio = torch.exp(log_ratio)

        # Get baseline
        value = linear_baseline(obs, reward, mask)

        # Get loss with baseline
        return_ = get_return(reward, mask)
        loss = -torch.mean(torch.sum(ratio * (return_ - value), dim=1))
        kl = kl_divergence(pi, old_pi).mean()

    return loss, kl, old_pi


def hessian_vector_product(model, kl, damping=1e-2):
    grads = torch.autograd.grad(kl, model.submodel.parameters_all[2](), create_graph=True)
    flat_grad_kl = parameters_to_vector(grads)

    def _product(vector, retain_graph=True):
        grad_kl_v = torch.dot(flat_grad_kl, vector)
        grad2s = torch.autograd.grad(grad_kl_v, model.submodel.parameters_all[2](), retain_graph=retain_graph)
        flat_grad2_kl = parameters_to_vector(grad2s)
        return flat_grad2_kl + damping * vector
    return _product


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()


def vector_to_parameters(vector, parameters):
    from torch.nn.utils.convert_parameters import _check_param_device

    param_device = None

    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer + num_param]
                         .view_as(param).data)

        pointer += num_param


def optimize(model, data, level, max_iter, optimizer, reset):      
    param_all = model.submodel.parameters_all

    if reset:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad=True)
        optimizer = SGD
        optim = optimizer([param_all[level]], lr=lr)
        optim = higher.get_diff_optim(optim, [param_all[level]])
    else:
        optim = optimizer(param_all[level](), lr=lr)

    for _ in range(max_iter):
        loss, memory = model(data, level)
        hierarchical_memory.add(memory)

        if reset:
            param_all[level], = optim.step(loss, params=[param_all[level]])
        else:
            print("[INFO] Length of memory:", len(hierarchical_memory.memories))
            old_loss, old_kl, old_pi = surrogate_loss(memory, model, old_pi=None)
            grads = torch.autograd.grad(old_loss, param_all[level](), retain_graph=True)
            grads = parameters_to_vector(grads)

            hessian = hessian_vector_product(model=model, kl=old_kl)
            stepdir = conjugate_gradient(hessian, grads)

            # Compute the Lagrange multiplier
            shs = 0.5 * torch.dot(stepdir, hessian(stepdir, retain_graph=False))
            lagrange_multiplier = torch.sqrt(shs / 1e-3)

            step = stepdir / lagrange_multiplier

            # Save the old parameters
            old_params = parameters_to_vector(model.submodel.parameters_all[2]())

            # Line search
            step_size = 1.0
            for _ in range(10):
                vector_to_parameters(old_params - step_size * step, model.submodel.parameters_all[2]())
                losses, kls, _ = surrogate_loss(memory, model, old_pi=old_pi)
                improve = losses - old_loss
                kl = kls
                if (improve.item() < 0.0) and (kl.item() < 1e-3):
                    break
                step_size *= 0.5

            else:
                vector_to_parameters(old_params, model.submodel.parameters_all[2]())

            hierarchical_memory.clear()

    print("Finished optimizing level", level, "\n")


def optimize_for_context(model, data, level, max_iter, optimizer, reset):      
    param_all = model.submodel.parameters_all

    if reset:
        param_all[level] = torch.zeros_like(param_all[level], requires_grad=True)
        optimizer = SGD
        optim = optimizer([param_all[level]], lr=lr)
        optim = higher.get_diff_optim(optim, [param_all[level]])

    for _ in range(max_iter):
        loss, memory = model.adapt(data, level)

        if reset:
            param_all[level], = optim.step(loss, params=[param_all[level]])
        else:
            return

    print("[ADAPT] Finished optimizing level", level, "\n")


basemodel = BaseModel(n_ctx, *layers)
model = Hierarchical_Model(basemodel) 
model(data, optimizer=Adam, reset=False)
