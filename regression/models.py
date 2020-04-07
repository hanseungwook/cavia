import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
import IPython

class Active_Weight(nn.Module):
    def __init__(self, n_input, n_output, n_context, gain_active=1, gain_passive=1, passive=True):  # n_bottleneck
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.w_passive, self.b_passive = 0, 0

        if passive:
            w, b = self.initialize(n_input, n_output, gain_passive)
            self.w_passive = Parameter(w.view(-1))
            self.b_passive = Parameter(b)

        if n_context > 0:
            w_all, b_all = [], []
            for i in range(n_context):
                w, b = self.initialize(n_input, n_output, gain_active)
                w_all.append(w)
                b_all.append(b)
            self.w_active = Parameter(torch.stack(w_all, dim=2).view(-1, n_context))
            self.b_active = Parameter(torch.stack(b_all, dim=1))

    def initialize(self, in_features, out_features, gain):
        weight = torch.Tensor(out_features, in_features)
        bias = torch.Tensor(out_features)
        self.reset_parameters(weight, bias)
        return gain * weight, gain * bias
    
    def reset_parameters(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
            
    def forward(self, context):
        batch_size = context.shape[0]
        if batch_size > 1:
            weight = F.linear(context, self.w_active, self.w_passive).view(batch_size, self.n_output, self.n_input)
            bias = F.linear(context, self.b_active, self.b_passive).view(batch_size, self.n_output)
        else: 
            weight = F.linear(context, self.w_active, self.w_passive).view(self.n_output, self.n_input)
            bias = F.linear(context, self.b_active, self.b_passive).view(self.n_output)
        
        return weight, bias


class Linear_Active(nn.Module):
    def __init__(self, n_input, n_output, n_context, gain_w, gain_b, passive): 
        super().__init__()
        self.active_weight = Active_Weight(n_input, n_output, n_context, gain_w, gain_b, passive)
        
    def forward(self, x, context):
        weight, bias = self.active_weight(context)
        batch_size = context.shape[0]
        
        if batch_size > 1:
            x = torch.einsum('bs,bas->ba', x, weight) + bias
        else:
            x = F.linear(x, weight, bias)
        
        return x


class Model_Active(nn.Module):
    def __init__(self, n_arch, n_context, gain_w = 1, gain_b = 1, nonlin=nn.ReLU(), passive=True, device=None): 
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = 'cpu'

        self.nonlin = nonlin
        self.depth = len(n_arch) - 1
        self.n_context = n_context
        # self.context_params = None
        # self.reset_context_params()
        
        module_list = []
        for i in range(self.depth):
            module_list.append(
                Linear_Active(n_arch[i], n_arch[i + 1], n_context, gain_w, gain_b, passive))

        self.module_list = nn.ModuleList(module_list)

    def reset_context(self):
        context = torch.zeros([1, self.n_context]).to(self.device)
        context.requires_grad = True
        return context

        
    def forward(self, x, context):
        for i, module in enumerate(self.module_list):
            x = module(x, context)
            if i < self.depth - 1:
                x = self.nonlin(x)
        return x



class Onehot_Encoder(nn.Module):
    def __init__(self, n_context, n_task): 
        super().__init__()
        self.linear = nn.Linear(n_task, n_context, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()  # Initialize to zero
    
    def reinit(self, n_context, n_task):
        self.linear = nn.Linear(n_task, n_context, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_() 
            
    def forward(self, input):
        context = self.linear(input)
    
        return context


class Encoder_Decoder(nn.Module):
#     def __init__(self, n_arch, n_context, n_task, gain_w, gain_b, decoder = None): 
    def __init__(self, encoder, decoder): 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reset_context(self):
        self.encoder.reinit_linear()

    def forward(self, inputs, encoder_input):
        context = self.encoder(encoder_input)
        pred = self.decoder(inputs, context)

        return pred

class Encoder_Decoder_VAE(nn.Module):
#     def __init__(self, n_arch, n_context, n_task, gain_w, gain_b, decoder = None): 
    def __init__(self, encoder, decoder): 
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reset_context(self):
        pass

    def forward(self, inputs, encoder_inputs):
        context, mu, logvar = self.encoder(encoder_inputs)
        
        # Repeating context number of batch_size, for each data point in each task
        batch_size = encoder_inputs.shape[1]
        num_context = context.shape[1]
        context = context.unsqueeze(1).repeat(1, batch_size, 1).reshape(-1, num_context)

        pred = self.decoder(inputs, context)

        return pred, mu, logvar

class Encoder_Core(nn.Module):
    def __init__(self, input_dim, n_hidden):
        super().__init__()
        self.input_dim = input_dim
        self.n_hidden  = n_hidden # 64

        self.fc1 = nn.Linear(input_dim, 9 * n_hidden)
        self.fc2 = nn.Linear(9 * n_hidden, 3 * n_hidden)
        self.fc3 = nn.Linear(3 * n_hidden, n_hidden)

    def _reshape_batch(self, input):
        return input.view(-1, input.shape[-1])
    
    def _mean_over_batch(self, input, mean_num, n_task):
        return input.view(-1, mean_num, input.shape[1]).mean(dim=1, keepdim=False).view(n_task,-1,input.shape[1])
        
    def _progressive_mean_over_batch(self, input, n_batch, n_batch_log, n_task, progressive):
        if progressive:
            temp = [self._mean_over_batch(input, 2**n, n_task) for n in range(n_batch_log+1)]
            temp = torch.cat(temp[::-1], dim=1)
        else: 
            temp = self._mean_over_batch(input, n_batch, n_task)
        return self._reshape_batch(temp) 

    def forward(self, input, progressive = False):
        n_task, n_batch, _ = input.shape        
        n_batch_log = int(torch.tensor(n_batch).float().log2())
        
        
        x = self._reshape_batch(input)
        x = F.relu(self.fc1(x))
        x = self._progressive_mean_over_batch(x, n_batch, n_batch_log, n_task, progressive)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

# Variational Encoder
class Encoder_Variational(nn.Module):
    def __init__(self, input_dim, n_context, n_hidden, progressive = False):
        super().__init__()
        self.progressive = progressive
        self.core = Encoder_Core(input_dim, n_hidden)
        self.fc3_mu = nn.Linear(n_hidden, n_context)
        self.fc3_var = nn.Linear(n_hidden, n_context)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, input, progressive = None):
        x = self.core(input, self.progressive) if progressive == None else self.core(input, progressive)
        mu = self.fc3_mu(x)
        logvar = self.fc3_var(x)

        if self.training:
            z = self._reparameterize(mu, logvar)
        else:
            z = mu

        return z, mu, logvar



# class Model_Test(nn.Module):
#     def __init__(self, model_decoder, n_task, n_context): 
#         super().__init__()
#         self.decoder = model_decoder
#         self.encoder = Onehot_Encoder(n_task, n_context)
        
# #         self.n_task = n_task
# #         self.n_context = n_context
# #         self.context = Parameter(torch.zeros([n_task, n_context]))

# #         assert n_task == 1, "Not implemented yet"
        
#     def forward(self, input, onenot):
# #         n_batch = input.shape[0]
# #         context = self.context.repeat(n_batch, 1)
#         context = self.encoder(onehot)
#         out = self.decoder(input, context)
#         return out

#     def reset_context(self):
#         for i_task in range(self.n_task):
#             for i_context in range(self.n_context):
#                 self.context[i_task, i_context].data.fill_(0)




class CaviaModel(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 n_arch,
                 n_context,
                 device
                 ):
        super(CaviaModel, self).__init__()

        self.device = device

        # fully connected layers
        self.fc_layers = nn.ModuleList()
        for k in range(len(n_arch) - 1):
            self.fc_layers.append(nn.Linear(n_arch[k], n_arch[k + 1]))

        # context parameters (note that these are *not* registered parameters of the model!)
        self.n_context = n_context
        # self.context_params = None
        # self.reset_context_params()

    def reset_context(self):
        context = torch.zeros(self.n_context).to(self.device)
        context.requires_grad = True
        return context

    def forward(self, x, context):

        # concatenate input with context parameters
        x = torch.cat((x, context.expand(x.shape[0], -1)), dim=1)

        for k in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)

        return y