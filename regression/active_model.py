import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init


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
        weight = F.linear(context, self.w_active, self.w_passive).view(self.n_output, self.n_input)
        bias = F.linear(context, self.b_active, self.b_passive).view(self.n_output)
        return weight, bias


class Linear_Active(nn.Module):
    def __init__(self, n_input, n_output, n_context, gain_w, gain_b, passive): 
        super().__init__()
        self.active_weight = Active_Weight(n_input, n_output, n_context, gain_w, gain_b, passive)
        
    def forward(self, x, context):
        weight, bias = self.active_weight(context)
        # print('x shape: {}'.format(x.shape))
        # print('weight shape: {}'.format(weight.shape))
        # x = torch.einsum('bs,as->ba', x, weight) + bias
        x = F.linear(x, weight, bias)
        return x


class Model_Active(nn.Module):
    def __init__(self, n_arch, n_context, gain_w, gain_b, nonlin=nn.ReLU(), passive=True, device=None): 
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = 'cpu'

        self.nonlin = nonlin
        self.depth = len(n_arch) - 1
        self.context_params = None
        self.n_context = n_context
        self.reset_context_params()
        # self.n_task = n_task
        
        module_list = []
        for i in range(self.depth):
            module_list.append(
                Linear_Active(n_arch[i], n_arch[i + 1], n_context, gain_w, gain_b, passive))

        self.module_list = nn.ModuleList(module_list)
    
    def reset_context_params(self):
        self.context_params = torch.zeros([1, self.n_context]).to(self.device)
        self.context_params.requires_grad = True
        
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
        
    def forward(self, input):
        context = self.linear(input)
        return context


class Encoder_Decoder(nn.Module):
#     def __init__(self, n_arch, n_context, n_task, gain_w, gain_b, decoder = None): 
    def __init__(self, decoder, n_context, n_task): 
        super().__init__()

        self.encoder = Onehot_Encoder(n_context, n_task)
        self.decoder = decoder

    def forward(self, input, onehot):
        context = self.encoder(onehot)
        pred = self.decoder(input, context)

        return pred

    

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
