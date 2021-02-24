import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.optim import Adam, SGD
import numpy as np
from functools import partial

from utils import move_to_device #, get_args  #, vis_img_recon #manual_optim, send_to
# from finite_diff import debug_top
# from torch.autograd import gradcheck

from torch.nn.utils.clip_grad import clip_grad_value_
# import higher 

from pdb import set_trace
import IPython

print_forward_test = False
print_forward_return = False #True
# print_task_loader = True
print_optimize_level_iter = False #True #
print_optimize_level_over = False

task_merge_len = 10 #5

##############################################################################
# TO DO: 
# add RNN encoder 
# add RL task & RL encoder 
# To Do: explort train and test loss: (line 55)


##############################################################################
#  Model Hierarchy

class Hierarchical_Model(nn.Module):            # Bottom-up hierarchy
    def __init__(self,  #levels, 
                        decoder_model, encoder_model, base_loss, logger, 
                        n_contexts, max_iters, for_iters, lrs, 
                        test_intervals,
                        log_loss_levels, log_ctx_levels, task_separate_levels, print_loss_levels,
                        Higher_flag = False, data_parallel=False): 
        
        super().__init__()

        if data_parallel:
            decoder_model  = nn.DataParallel(decoder_model)
        self.decoder_model = decoder_model            
        self.encoder_model = encoder_model
        self.base_loss = base_loss
        # self.adaptation = optimize if encoder_model is None else encoder_model 
        self.logger = logger

        # self.level_max   = levels
        self.n_contexts  = n_contexts
        self.max_iters   = max_iters
        self.for_iters   = for_iters
        self.lrs         = lrs 
        self.test_intervals = test_intervals

        self.log_loss_levels    = log_loss_levels
        self.log_ctx_levels     = log_ctx_levels
        self.task_separate_levels = task_separate_levels
        self.print_loss_levels  = print_loss_levels
        self.Higher_flag        = Higher_flag 
        self.device             = decoder_model.device
        

    def forward(self, level, task_list, 
                        optimizer=SGD, reset=True, return_outputs=False, 
                        status='', #current_status='', 
                        viz=None, iter_num = None):        
        '''
        Compute average loss over multiple tasks in task_list
        returns:  mean_loss, outputs

        Encoder(Adaptation) + Decoder model:
        '''
                
# lv3: root: 1 env. static (same as lv2 loss)
# lv2: mean(sine + line)/2
# lv1: mean sines
# lv0: mean pixel (1 sine)
        if status != '':
            status += '_lv'+str(level)

        if level > 0:      # meta-level evaluation
            task_merge_flag = (level-1) in self.task_separate_levels #[level-1] #len(task_list) > task_merge_len
            meta_eval_partial = partial(self.meta_eval, level = level-1, status =  status, optimizer=optimizer, reset=reset, return_outputs=return_outputs, iter_num=iter_num, task_merge_flag=task_merge_flag)
            loss, outputs = get_average_loss(meta_eval_partial, task_list, return_outputs) 

#             if status == viz:   #visualize_output(outputs)

        else:                  # base-level evaluation (level 0)
            inputs, targets, _ = move_to_device(task_list, self.device)
            assert isinstance(inputs, (torch.FloatTensor, torch.DoubleTensor, torch.cuda.FloatTensor, torch.cuda.DoubleTensor))
            outputs = self.decoder_model(inputs)
            loss = self.base_loss(outputs, targets)

        # assert not torch.isnan(loss), "loss is nan"
        if torch.isnan(loss):
            print("loss is nan")
            set_trace()
        
        if status != '' and (level in self.log_loss_levels):
            self.log_loss(loss.item(), status, iter_num, print=(level in self.print_loss_levels))              #log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))
        return loss, outputs
    

    def meta_eval(self, task, task_idx, level, status, 
                        optimizer, reset, return_outputs, iter_num, task_merge_flag):
        '''  Applies adaptation on train-tasks and then evaluates the generalization loss on test-tasks     '''
        if not task_merge_flag and status != '':

            status += '/task' + str(task_idx)  # current_status = 'lv'+str(level)+'_task'+str(task.idx)

        def run_test(iter_num):       # evaluate on one mini-batch from test-loader
            sample_type = 'test'
            l, outputs_ = self.forward(level, task.load(sample_type), return_outputs=return_outputs, status=status+'/'+sample_type, iter_num=iter_num) #, current_status = 'test' + current_status)    # test only 1 minibatch
            return l, outputs_

        # inner-loop optimization
        sample_type = 'train' 
        optimize(self, task.load(sample_type), level,
                        self.lrs[level], self.max_iters[level], self.for_iters[level], #self.args_dict, 
                        optimizer = optimizer, reset = reset, 
                        status = status+'/'+sample_type, 
                        # current_status =  'train' + current_status, 
                        run_test = run_test, test_interval = self.test_intervals[level],
                        device = self.device, Higher_flag = self.Higher_flag, 
                        # log_loss_flag = (level in self.log_loss_levels),
                        log_ctx_flag = (level in self.log_ctx_levels))

        test_loss_opt = run_test(iter_num=self.max_iters[level]) # final test-loss
        return test_loss_opt

        
    #################
    ####  Logging 
    def log_loss(self, loss, status, iter_num, print=False): #, name):
        self.logger.experiment.add_scalar("loss{}".format(status), loss, iter_num)   #  .add_scalars("loss{}".format(status), {name: loss}, iter_num)

        if print:
            print('Loss {} Itr {}'.format(status, iter_num))

    def log_ctx(self, ctx, status, iter_num):   #   def log_ctx(self, status, current_status, ctx):
        if ctx is None or ctx.numel() == 0 or self.logger is None:
            pass
        else:      # Log each context changing separately if size <= 3
            if ctx.numel() <= 4:
                for i, ctx_ in enumerate(ctx.flatten()): #range(ctx.size):
                    self.logger.experiment.add_scalar("ctx{}".format(status,i), ctx_, iter_num)
                    # self.logger.experiment.add_scalar("ctx{}/{}".format(status,i), {current_status: ctx_}, iter_num)
            else:
                self.logger.experiment.add_histogram("ctx{}".format(status), ctx, iter_num)


##################################
def get_average_loss(model, task_list, return_outputs):
    loss, outputs = [], []

    for (param, task, idx) in task_list:   # Todo: Parallelize! see SubprocVecEnv: https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html
        l, outputs_ = model(task, idx)
        loss.append(l)
        if return_outputs:
            outputs.append(outputs_)

    return torch.stack(loss).mean() , outputs






##########################################################################################################################################   
# optimize
##########################################################################################################################################   

def optimize(model, dataloader, level, 
            lr, max_iter, for_iter, optimizer, reset, 
            status, # current_status, 
            run_test, test_interval, 
            device, Higher_flag, 
            log_ctx_flag):  #log_loss_flag, 
    ## optimize parameter for a given 'task'
    grad_clip_value = 100 #1000
    # print(level)
    
    ################
    def initialize():     ## Initialize param & optim
        param_all = model.decoder_model.module.parameters_all if isinstance(model.decoder_model, nn.DataParallel) else model.decoder_model.parameters_all

        optim = None
        if param_all[level] is not None:
            if reset:  # use manual optim or higher 
                param_all[level] = torch.zeros_like(param_all[level], requires_grad=True, device=device)   # Reset
                if Higher_flag: # using higher 
                    optim = optimizer([param_all[level]], lr=lr)
                    optim = higher.get_diff_optim(optim, [param_all[level]]) #, device=x.device) # differentiable optim for inner-loop:
            else: # use regular optim: for outer-loop
                optim = optimizer(param_all[level](), lr=lr)   # outer-loop: regular optim
                
        return param_all, optim

    def update_step():
        if param_all[level] is not None:      
            if reset:  # use manual SGD or higher 
                if Higher_flag:
                    param_all[level], = optim.step(loss, params=[param_all[level]])    # syntax for diff_optim
                else: # manual SGD step
                    first_order = False
                    if param_all[level+1] is None:  # for LQR_.. without network model parameters
                        first_order = True
                        
                    grad = torch.autograd.grad(loss, param_all[level], create_graph=not first_order)
                    for g in grad: #filter(lambda p: p.grad is not None, parameters):
                        g.data.clamp_(min=-grad_clip_value, max=grad_clip_value)
                    param_all[level] = param_all[level] - lr * grad[0]

            else: # use regular optim: for outer-loop
                optim.zero_grad()
                loss.backward()
                clip_grad_value_(param_all[level](), clip_value=grad_clip_value) #20)
                optim.step()  
                
#     def logging(loss, param, cur_iter):
# #         if not (cur_iter % update_iter[level]):
#             # if log_loss_flag:
#             #     model.log_loss(loss, status+current_status, cur_iter)  #log[self.log_name].info("At iteration {}, meta-loss: {:.3f}".format(self.iter, loss))
#             if log_ctx_flag:
#                 model.log_ctx(param, status + current_status, cur_iter)  #  log the adapted ctx for the level
                
    ######################################
    # main code
    # task_idx = dataloader.task_idx if hasattr(dataloader,'task_idx') else None
    # task_name = dataloader.task_name if hasattr(dataloader,'task_name') else None
    
    param_all, optim = initialize()  
    cur_iter = 0
    loss = None
    # status = status # + current_status
    while True:
        for i, task_batch in enumerate(dataloader):
            for _ in range(for_iter):          # Seungwook: for_iter is to replicate cavia’s implementation where they use the same mini-batch for the inner loop steps
                if cur_iter >= max_iter:       # Terminate! 
                    # if print_optimize_level_over and loss is not None:
                    #     print('optimize '+ status, 'loss_final', loss.item())
                    return None   # param_all[level]    # for log_ctx

                loss, output = model.forward(level, task_batch, status = status, #current_status = current_status, 
                                                                iter_num = cur_iter)    # Loss to be optimized
                update_step()

                if log_ctx_flag and reset:
                    model.log_ctx(param_all[level], status + '_lv'+str(level), cur_iter)  #  log the adapted ctx for the level
                # logging(loss.item(), param_all[level], cur_iter)


                # Run Test-loss
                if not (cur_iter % test_interval) and cur_iter <= max_iter - test_interval:
                    test_loss, test_outputs = run_test(iter_num = cur_iter)  # get test_loss 
                    if level>0:
                        print('level',level, 'cur_iter', cur_iter, 'test loss', test_loss.item())

                cur_iter += 1  