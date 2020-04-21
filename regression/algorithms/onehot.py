import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
from algorithms.base import Base
from train import get_model_decoder, get_model_encoder, create_targets, create_inputs_targets, create_1hot_labels
from models import Encoder_Decoder


class OneHot(Base):
    def __init__(self):
        super(OneHot, self).__init__()
        # self.model = None
        
    def build_model(self, args, task_family):
        model_encoder = get_model_encoder(args, task_family['train'])
        model_decoder = get_model_decoder(args, task_family['train'])
        
        self.model = Encoder_Decoder(model_encoder, model_decoder)

        return self.model

    def eval_model(self, inputs, targets, labels_1hot, cur_model=None):
        if cur_model:
            model = cur_model
        else:
            model = self.model

        outputs = model(inputs, labels_1hot)
    
        return F.mse_loss(outputs, targets)
    
    def eval_model_total(self, args, task_family, num_updates, n_tasks=25, task_type='train'):    
        # logging
        model = copy.deepcopy(self.model)

        losses = []

        # Optimizer for encoder (context)
        inner_optimizer = optim.SGD(model.encoder.parameters(), args.lr_inner)

        # Sample tasks
        target_functions, labels_1hot = task_family.sample_tasks_1hot(total_num_tasks=n_tasks*4, batch_num_tasks=n_tasks, batch_size=args.k_shot_eval)

        # Only if task is valid or test, reinitialize model and perform inner updates
        if task_type != 'train':
            # Reinitialize one hot encoder for evaluation mode
            model.encoder.reinit(args.num_context_params, task_family.total_num_tasks)
            model.encoder.train()
            
            # Inner loop steps
            for _ in range(1, num_updates + 1):
                inputs = task_family.sample_inputs(args.k_meta_test * n_tasks, args.use_ordered_pixels).to(args.device)
                targets = create_targets(inputs, target_functions, n_tasks)
                labels = create_1hot_labels(labels_1hot, n_tasks)

                loss = self.update_step(inner_optimizer, inputs, targets, labels, n_tasks, cur_model=model)

        # Get entire range's input and respective 1 hot labels
        input_range = task_family.get_input_range().to(args.device).repeat(n_tasks, 1)
        bs = int(input_range.shape[0] / n_tasks)
        input_range_1hot = torch.cat([task_family.create_input_range_1hot_labels(batch_size=bs, cur_label=labels_1hot[t, 0, :]) for t in range(n_tasks)])
        input_range_targets = create_targets(input_range, target_functions, n_tasks)
        
        # compute true loss on entire input range
        model.eval()
        losses.append(self.eval_model(input_range, input_range_targets, input_range_1hot, cur_model=model).item())
        #self.viz_pred(model, input_range[:bs], input_range_targets[:bs], input_range_1hot[:bs], task_type)
        model.train()

        losses_mean = np.mean(losses)

        return losses_mean

    def meta_backward(self, args, inner_optimizer, outer_optimizer, task_family, target_functions, labels_1hot):
        # --- compute meta gradient ---    
        train_inputs = task_family['train'].sample_inputs(args.k_meta_train * args.tasks_per_metaupdate, args.use_ordered_pixels).to(args.device)
        test_inputs = task_family['test'].sample_inputs(args.k_meta_train * args.tasks_per_metaupdate, args.use_ordered_pixels).to(args.device)

        # Create targets and labels of all tasks in batch level
        train_targets = create_targets(train_inputs, target_functions, args.tasks_per_metaupdate)
        test_targets = create_targets(test_inputs, target_functions, args.tasks_per_metaupdate)
        labels = create_1hot_labels(labels_1hot, args.tasks_per_metaupdate)
        
        # Encoder / inner update
        self.update_step(inner_optimizer, train_inputs, train_targets, labels, args.tasks_per_metaupdate)

        # Decoder / outer update
        self.update_step(outer_optimizer, test_inputs, test_targets, labels, args.tasks_per_metaupdate)

    def update_step(self, optimizer, inputs, targets, labels, n_tasks, cur_model=None):
        # Update given optimizer (whether inner or outer)
        optimizer.zero_grad()

        loss = self.eval_model(inputs, targets, labels, cur_model)
        loss /= n_tasks
        loss.backward()

        optimizer.step()

        return loss
    
    def viz_pred(self, inputs, targets, labels, task_type=None):
        pass
