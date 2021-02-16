import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_value_

from pdb import set_trace

def LQR_tasks(x_dim, batch_nums, x_range = 4, kbm_zero=False):

    batch_lv0, batch_lv1, batch_lv2 = batch_nums

    #level 2 - dynamics
    kbm = (0.,0.,0.) if kbm_zero  else (0.0*np.random.randn(batch_lv2), 0.5*np.random.randn(batch_lv2), 0.5*np.random.randn(batch_lv2))

    #level 1 - targets
    target   = x_range * torch.randn(1, batch_lv1*batch_lv2)   # Delete this
#     target   = x_range * torch.randn(1, batch_lv1, batch_lv2)
    
    #level 0 - initial locations
    x0       = torch.zeros(x_dim, batch_lv0*batch_lv1*batch_lv2)   # Delete this
    x0[0,:]  = x_range*torch.randn(1, batch_lv0*batch_lv1*batch_lv2)   # Delete this
#     x0       = torch.zeros(x_dim, batch_lv0, batch_lv1, batch_lv2)
#     x0[0,:]  = x_range*torch.randn(1, batch_lv0, batch_lv1, batch_lv2)

    return kbm, target, x0


def make_AB(x_dim, kbm): # = (0.,0.,0.)):

    dt = 1.0
    if x_dim == 1:
        A = [[0.0]]
    elif x_dim == 2:
        A = [[0,1],[0,0]]
    elif x_dim == 3:
        A = [[0,1,0],[0,0,1],[0,0,0]]
    
    
    A_kbm = x_dim*[x_dim*[0]]; A_kbm[-1] = kbm[:x_dim]       # j + ma + bv + kx = F  #   k, b, m = kbm 
    A_tensor = torch.eye(x_dim) + (torch.tensor(A).float() - torch.tensor(A_kbm).float()) * dt

    B = x_dim*[[0]]; B[-1] = [1]
    B_tensor = torch.tensor(B).float() * dt 
    
    return A_tensor, B_tensor


class LQR_environment():

#     def __init__(self, x_dim, batch_nums, x_range = 4, kbm = (0.,0.,0.)):
#         self.x_dim = x_dim
#         self.batch_nums = batch_nums
#         batch_lv0, batch_lv1, batch_lv2 = batch_nums
#         self.batch_all = batch_lv0*batch_lv1*batch_lv2  # Delete this
#         self.x_range = x_range 
#         self.kbm = kbm
#         self.task0 = LQR_tasks(self.x_dim, self.batch_nums, self.x_range, kbm_zero = False)
    
#     def make_env(self, task = None):

    def __init__(self, task, x_dim=2):
        
#         if task is None:
#             task = LQR_tasks(self.x_dim, self.batch_nums, self.x_range, kbm_zero)
        kbm, target, x0 = task   #      A, B, target, x0 = task
        # kbm and target are collated by dataloader.. all repeated. so just take [0] element
        kbm    = kbm[0].float() # [k[0].float() for k in kbm] #[:x_dim]
        target = target[0].float()
        
        x0    = x0.view(1,-1).float()
        batch = x0.shape[1]
        x0    = torch.cat([x0,torch.zeros(x_dim-1, batch)], dim=0)
        
        
        A, B = make_AB(x_dim, kbm)
#         print('kbm', kbm, 'target', target, 'x0', x0)
#         print('A', A)
#         print('B', B)

        def dynamics(x,u):
            return A @ x + B @ u  #  x + (A @ x + B * u) * dt

        def cost_fnc(x,u):
            target_cost = ((x[0,:]-target)**2).mean() / 3 #if sq_loss else 1 - (-pos**2).exp()
            return target_cost + (u**2).mean()

        self.dynamics = dynamics #dyn_gen(A, B)
        self.cost_fnc = cost_fnc     #cost_gen(target)
        self.A, self.B, self.target = A, B, target
        self.state    = x0
        
#         return self.observe() 
        
    def observe(self):
        return self.state  # full state observation
    
    def step(self, u):
        loss = self.cost_fnc(self.state, u)
        self.state = self.dynamics(self.state, u)
        return loss, self.observe() #self.state
    

class Linear_Policy(nn.Module):
    def __init__(self, x_dim): #(ab,target):
        super().__init__()
        self.x_dim = x_dim
#         self.ctx_target = nn.Parameter(torch.zeros(1)) #  Level 0 ctx
#         self.ctx_coeff = nn.Parameter(torch.zeros(x_dim).t()) #  Level 1 ctx
#         self.ctx_target = nn.Parameter(torch.randn(1))  #  Level 0 ctx
#         self.ctx_coeff = nn.Parameter(torch.randn(x_dim).t())  #  Level 1 ctx
#         self.parameters_all = [self.ctx_target,  self.ctx_coeff, None] 
        
#     def __init__(self, x_dim, batch_nums): #(ab,target):
#         super().__init__()
#         batch_lv0, batch_lv1, batch_lv2 = batch_nums
        
#         self.ctx_coeff = nn.Parameter(torch.zeros(x_dim).t()) #, requires_grad = True))
#         self.ctx_target = nn.Parameter(torch.zeros(1,batch_lv1*batch_lv2)) # nn.Parameter(torch.zeros(1)) 
#         self.parameters_all = [self.ctx_target,  self.ctx_coeff] 
        
    def forward(self, obs, t, ctx):  # action = function ( observation, time )
#         ctx = torch.cat(ctx)
        ctx = ctx[0]
        ctx_target, ctx_coeff = ctx[:1], ctx[1:] 
        action = ctx_coeff[:1] @ ( obs[:1,:] - ctx_target )   #         action = self.coeff @ ( x - self.target ) 
        if ctx_coeff.shape[0]>1:
            action += ctx_coeff[1:] @ obs[1:,:]
        return action.view(1,-1)


# class Action_trajectory(nn.Module):
#     def __init__(self, T, batch_nums): #(ab,target):
#         super().__init__()
#         batch_lv0, batch_lv1, batch_lv2 = batch_nums
        
#         self.u_all = nn.Parameter(torch.zeros(T,batch_lv0*batch_lv1*batch_lv2))  # Delete this 
# #         self.u_all = nn.Parameter(torch.zeros(T,batch_lv0, batch_lv1, batch_lv2))
#     def forward(self, obs, t):
#         action = self.u_all[t,:]
#         return action
    
def make_ctx(n, device):
    # if DOUBLE_precision:
    #     return torch.zeros(1,n, requires_grad=True).double()
    # else:
        return torch.zeros(1,n, requires_grad=True, device=device)
      


class Combine_NN_ENV(nn.Module):
    device = 'cpu'
    
    def __init__(self, policy = None, x_dim=2, levels = None, batch_nums = (1,3,1), kbm_zero = True, random_task = False, T = 12, traj_opt = False, x_range = 4):
        super().__init__()
        
#         n_ctx = [1,2]
        
        ctx_target = torch.zeros(1, device=self.device) 
        ctx_coeff = torch.zeros(x_dim, device=self.device)   
        if levels == 2:  #         for "LQR_lv2"
#             self.parameters_all = [ctx_target, ctx_coeff, None] # [self.layers.parameters]
            self.parameters_all = [torch.cat([ctx_target, ctx_coeff]), None, None] # [self.layers.parameters]
        elif levels == 1:  #         for "LQR_lv1"
            self.parameters_all = [torch.cat([ctx_target, ctx_coeff]), None] # [self.layers.parameters]
        else:
            error()
            
#         if policy is None:
#             if traj_opt:
#                 assert random_task is False
#                 policy = Action_trajectory(T, batch_nums) 
#             else:
#                 policy = Linear_Policy(x_dim, batch_nums)
        policy = Linear_Policy(x_dim)
    
        self.policy = policy
#         self.parameters_all = policy.parameters_all
#         self.parameters_all = [policy.ctx_target,  policy.ctx_coeff, None] 
        
        self.T = T
#         self.env = LQR_environment(x_dim, batch_nums, x_range = x_range) #, kbm=kbm)
#         self.random_task = random_task
#         self.traj_opt = traj_opt
        
#         self.obs0 = None if random_task else self.env.make_env(self.env.task0)            

    def forward(self, task, record = True): #:
        
        ctx_all = self.parameters_all[:-1]
        
#         if self.random_task: #self.obs0 is None: #
#             obs = self.env.make_env(task)            
#         else:
#             obs, self.env.state = self.obs0, self.obs0
        self.env = LQR_environment(task) #, kbm=kbm)
        obs = self.env.observe()
            
        l = 0.0
        x_all, u_all = [], [];    
        for t in range(self.T):
            action = self.policy(obs, t, ctx_all)
            loss, obs = self.env.step(action)
            l = l + loss
            
            if record:  ## Change to logger.update()
                x_all.append(obs[0,:].detach().view(-1))  #position only
                u_all.append(action.detach().view(-1))
                    
#         set_trace()
        return l, (torch.stack(x_all).numpy(), torch.stack(u_all).numpy() )
    

#     def train_step(self, task, optim):
#         optim.zero_grad()
#         l, x, u = self.forward(task, record = True)
#         l.backward()
#         clip_grad_value_(self.policy.parameters(), clip_value=20)
#         optim.step()   
#         return l.item(), x, u
    
    
#     def main(self, epoch, lr, task = None, optim = None):
        
#         if self.traj_opt:
#                 assert self.random_task is False
# #             if optim is None:
#                 optim = torch.optim.SGD(self.policy.parameters(),  lr=lr, momentum = 0.9)
#         else:
# #             if optim is None:
#                 optim = torch.optim.SGD([{"params": self.policy.ctx_coeff},
#                                          {"params": self.policy.ctx_target, "lr": lr*self.env.batch_all*4}  ],
#                                          lr=lr, momentum = 0.9)

#         l_all, coeff_all, target_all  = [], [], []

#         for iter_ in range(epoch):
#             l, x, u = self.train_step(task, optim)
#             l_all.append(l)
        
#             if not self.traj_opt:
#                 coeff_all.append(self.policy.ctx_coeff.clone().detach())
#                 target_all.append(self.policy.ctx_target.squeeze().clone().detach())

#         if self.traj_opt:
#             return l_all, x, u, coeff_all, target_all, self.env.target.numpy(), self.policy, 
#         else:
#             return l_all, x, u, torch.stack(coeff_all).numpy(), torch.stack(target_all).numpy(), self.env.target.numpy(), self.policy, 
        
    
#############################


## LQR task

x_range = 4
def sample_LQR_LV2(sample_type):                      #level 2 - dynamics
    kbm =  np.stack([np.random.uniform(0,0), np.random.uniform(-0, 0.5),  np.random.uniform(-0, 0.5)], axis=0)
    def sample_LV1(sample_type):                      #level 1 - goal
        goal   = x_range * np.random.randn(1)
        def sample_LV0(batch_size, sample_type):      #level 0 - initial  x0
            pos0   = x_range * np.random.randn(batch_size)
            task_env = kbm, goal, pos0
            return task_env
        return sample_LV0, None    # returning input_sampler (and target_sampler = None) for level0
    return sample_LV1

def sample_LQR_LV1(sample_type):                         #level  - dynamics, goal
    kbm =  np.stack([np.random.uniform(0,0), np.random.uniform(-0, 0.5),  np.random.uniform(-0, 0.5)], axis=0)
    goal   = x_range * np.random.randn(1)
    def sample_LV0(batch_size, sample_type):      #level 0 - initial  x0
        pos0   = x_range * np.random.randn(batch_size)
        task_env = kbm, goal, pos0
        return task_env
    return sample_LV0, None    # returning input_sampler (and target_sampler = None) for level0

def sample_LQR_LV0(batch_size, sample_type):      #level 0 - initial  x0
    kbm =  np.stack([np.random.uniform(0,0), np.random.uniform(-0, 0.5),  np.random.uniform(-0, 0.5)], axis=0)
    goal   = x_range * np.random.randn(1)
    pos0   = x_range * np.random.randn(batch_size)
    task_env = kbm, goal, pos0
    return task_env



#############################

def plot_all(l_all, u_opt, x_opt, coeff_all, target_all, target0, traj_opt = False):
    plt.figure()
    plt.plot(u_opt)
    plt.plot(x_opt)
#     plt.legend(['control', 'position', 'velocity', 'acceleration'])
    plt.plot(target0.repeat(len(u_opt),axis=0),':')
    
    plt.show()

    plt.figure()   #     plt.subplot(1, 2, 1)
    plt.semilogy(l_all)
    plt.legend(['loss'])

    if not traj_opt:
        plt.figure()
        plt.plot(coeff_all)
        plt.plot(target_all) # - target0)
        plt.legend(['c1', 'c2', 'c3'][:len(coeff_all[0])] + ['tar1', 'tar2', 'tar3'])
        plt.plot(target0.repeat(len(coeff_all),axis=0),':')
        plt.show()
