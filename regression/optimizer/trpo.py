import torch
import numpy as np
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector
from misc.torch_utils import detach_distribution, vector_to_parameters
from misc.linear_baseline import LinearFeatureBaseline, get_return


def surrogate_loss(task, memory, model, old_pi=None):
    model.base_model.reset_context()
    model(task, reset=False, is_outer=True, optimizer=TRPO)
    ctx = model.base_model.parameters_all[:-1]

    with torch.set_grad_enabled(old_pi is None):
        obs, action, logprob, reward, mask = memory.sample()
        pi = model.base_model(torch.from_numpy(np.stack(obs, axis=1)).float(), ctx_=ctx)

        if old_pi is None:
            old_pi = detach_distribution(pi)

        action = torch.stack(action, dim=1)
        log_ratio = (pi.log_prob(action) - old_pi.log_prob(action))
        ratio = torch.exp(log_ratio)

        # Get baseline
        linear_baseline = LinearFeatureBaseline(obs)
        value = linear_baseline(obs, reward, mask)

        # Get loss with baseline
        return_ = get_return(reward, mask)
        loss = -torch.mean(torch.sum(ratio * (return_ - value), dim=1))
        kl = kl_divergence(pi, old_pi).mean()

    return loss, kl, old_pi


def hessian_vector_product(model, kl, damping=1e-2):
    grads = torch.autograd.grad(kl, model.base_model.parameters_all[2](), create_graph=True)
    flat_grad_kl = parameters_to_vector(grads)

    def _product(vector, retain_graph=True):
        grad_kl_v = torch.dot(flat_grad_kl, vector)
        grad2s = torch.autograd.grad(grad_kl_v, model.base_model.parameters_all[2](), retain_graph=retain_graph)
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


class TRPO(object):
    def __init__(self):
        pass

    def step(self, memories, model):
        param_all = model.base_model.parameters_all
    
        task = model.task
        old_losses, old_kls, old_pis = [], [], []
        for memory in memories:
            old_loss, old_kl, old_pi = surrogate_loss(task, memory[0], model, old_pi=None)
            old_losses.append(old_loss)
            old_kls.append(old_kl)
            old_pis.append(old_pi)
    
        old_loss = sum(old_losses) / float(len(old_losses))
        grads = torch.autograd.grad(old_loss, param_all[-1](), retain_graph=True)
        grads = parameters_to_vector(grads)
    
        old_kl = sum(old_kls) / float(len(old_kls))
        hessian = hessian_vector_product(model=model, kl=old_kl)
        stepdir = conjugate_gradient(hessian, grads)
    
        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / 1e-3)
    
        step = stepdir / lagrange_multiplier
    
        # Save the old parameters
        old_params = parameters_to_vector(model.base_model.parameters_all[2]())
    
        # Line search
        step_size = 1.0
        for _ in range(10):
            vector_to_parameters(old_params - step_size * step, model.base_model.parameters_all[2]())
    
            losses, kls = [], []
            for memory, old_pi in zip(memories, old_pis):
                loss, kl, _ = surrogate_loss(task, memory[0], model, old_pi=old_pi)
                losses.append(loss)
                kls.append(kl)
            loss = sum(losses) / float(len(losses))
            kl = sum(kls) / float(len(kls))
    
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < 1e-3):
                break
            step_size *= 0.5
    
        else:
            vector_to_parameters(old_params, model.base_model.parameters_all[2]())
