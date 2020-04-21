import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from algorithms.base import Base
from train import get_model_decoder, get_model_encoder, create_targets, create_inputs_targets
from models import Encoder_Decoder_VAE


class VAE(Base):
    def __init__(self):
        super(VAE, self).__init__()
        # self.model = None
        
    def build_model(self, args, task_family):
        model_encoder = get_model_encoder(args, task_family['train'])
        model_decoder = get_model_decoder(args, task_family['train'])
        self.model = Encoder_Decoder_VAE(model_encoder, model_decoder)

        return self.model

    def eval_model(self, inputs, targets, inputs_targets, writer, i_iter, tag, cur_model=None):
        if cur_model:
            model = cur_model
        else:
            model = self.model

        outputs, mu, logvar = model(inputs, inputs_targets)
        pred_loss = F.mse_loss(outputs, targets)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.001

        if writer:
            writer.add_scalar('Loss/{}_total'.format(tag), pred_loss + kld, i_iter)
            writer.add_scalar('Loss/{}_pred'.format(tag), pred_loss, i_iter)
            writer.add_scalar('Loss/{}_kld'.format(tag), kld, i_iter)
        # kld = torch.tensor(0)
    
        return pred_loss + kld
    
    def eval_model_total(self, args, task_family, n_tasks=25, task_type='train'):
        # logging
        losses = []

        # Sample tasks
        target_functions = task_family.sample_tasks_vae(total_num_tasks=n_tasks*4, batch_num_tasks=n_tasks, batch_size=args.k_shot_eval)

        # Get entire range's input and respective 1 hot labels
        input_range = task_family.get_input_range().to(args.device).repeat(n_tasks, 1)
        input_range_targets = create_targets(input_range, target_functions, n_tasks)

        bs = int(input_range.shape[0] / n_tasks)
        inputs_targets = create_inputs_targets(input_range, input_range_targets, n_tasks, bs)
        
        with torch.no_grad():
            # compute true loss on entire input range
            self.model.eval()
            losses.append(self.eval_model(input_range, input_range_targets, inputs_targets, None, None, None).item())
            # viz_pred(model, input_range[:bs], input_range_targets[:bs], inputs_targets[0].unsqueeze(0), task_type)
            self.model.train()

        losses_mean = np.mean(losses)

        return losses_mean

    def meta_backward(self, args, inner_optimizer, outer_optimizer, task_family, target_functions, writer, i_iter):
        # Sample inputs 
        train_inputs = task_family['train'].sample_inputs(args.k_meta_train * args.tasks_per_metaupdate, args.use_ordered_pixels).to(args.device)
        test_inputs = task_family['test'].sample_inputs(args.k_meta_train * args.tasks_per_metaupdate, args.use_ordered_pixels).to(args.device)

        # Create targets and labels of all tasks in batch level
        train_targets = create_targets(train_inputs, target_functions, args.tasks_per_metaupdate)
        test_targets = create_targets(test_inputs, target_functions, args.tasks_per_metaupdate)

        # Collate inputs and targets into shape (n_task, batch_size, input_dim + output_dim)
        train_inputs_targets = create_inputs_targets(train_inputs, train_targets, args.tasks_per_metaupdate, args.k_meta_train)
        test_inputs_targets = create_inputs_targets(test_inputs, test_targets, args.tasks_per_metaupdate, args.k_meta_train)

        # Encoder / inner update
        self.update_step(inner_optimizer, train_inputs, train_targets, train_inputs_targets, args.tasks_per_metaupdate, writer, i_iter, tag='inner')

        # Decoder / outer update
        self.update_step(outer_optimizer, test_inputs, test_targets, test_inputs_targets, args.tasks_per_metaupdate, writer, i_iter, tag='outer')


    def update_step(self, optimizer, inputs, targets, inputs_targets, n_tasks, writer, i_iter, tag):
        # Update given optimizer (whether inner or outer)
        optimizer.zero_grad()

        loss = self.eval_model(inputs, targets, inputs_targets, writer, i_iter, tag)
        loss /= n_tasks
        loss.backward()

        optimizer.step()

        return loss
    
    def viz_pred(self, inputs, targets, labels, task_type=None):
        with torch.no_grad():
            self.model.eval()
            preds = self.model(inputs, labels)
            self.model.train()
            
        if len(preds) == 3:
            preds = preds[0]
        
        plt.scatter(inputs.detach().numpy(), targets.detach().numpy(), c='green', label='Target', s=1)
        plt.scatter(inputs.detach().numpy(), preds.detach().numpy(), c='blue', label='Pred', s=1)
        plt.legend()

        if task_type:
            plt.title(task_type)
        plt.show()
