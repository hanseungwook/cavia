import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from functools import partial

# from pdb import set_trace

def get_encoder_type(encoder_type):
    if encoder_type == "HSML":
        MODEL = HSML
    else:
        raise ValueError()
    return ENCODER

def get_model_type(model_type):

    if model_type == "CAVIA":
        MODEL = Cavia
    elif model_type == "ADDITIVE":
        MODEL = Model_Additive
    elif model_type == "MULTIPLICATIVE":
        MODEL = Model_Multiplicative
    elif model_type == "ADD_MULTIPLICATIVE":
        MODEL = Model_Add_Multiplicative
    else:
        raise ValueError()

    return MODEL


class BaseModel(nn.Module):
    """     Feed-forward neural network with context    """
    def __init__(self, n_arch, n_context, nonlin, loss_fnc, device, FC_module):
        super().__init__()

        self.device = device if device is not None else 'cpu'
        self.n_context = n_context #sum(n_context)  # Concatenate all high-level ctx into one. 
        self.nonlin = nonlin
        self.loss_fnc = loss_fnc
        self.n_arch = n_arch
        
        self.module_list = nn.ModuleList()
        for i in range(len(n_arch) - 1):
            self.module_list.append(FC_module(n_arch[i], n_arch[i + 1]))    # Fully connected layers

    def evaluate(self, data, ctx):
        inputs, targets = data
        outputs = self.forward(inputs, ctx)
        return self.loss_fnc(outputs, targets)

######################################

class Cavia(BaseModel):
    def __init__(self, n_arch, n_context, nonlin=nn.ReLU(), loss_fnc = nn.MSELoss(), device = None):

        n_arch[0] += n_context  # add n_context to n_input 

        super().__init__(n_arch, n_context, nonlin, loss_fnc, device, FC_module = nn.Linear)


    def forward(self, x, ctx_list):
        ctx = torch.cat(ctx_list, dim=1)                  # combine ctx with higher-level ctx
        x = torch.cat((x, ctx.expand(x.shape[0], -1)), dim=1)   # Concatenate input with context
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i < len(self.module_list) - 1:  
                x = self.nonlin(x)
        return x

######################################

class Model_Active(BaseModel):
    def __init__(self, n_arch, n_context, weight_type, gain_w = 1, gain_b = 1, nonlin=nn.ReLU(), loss_fnc = nn.MSELoss(), passive=True, device=None): 
        if weight_type == 'additive':
            active_type = Additive_Weight
        elif weight_type == 'multiplicative':
            active_type = Multive_Weight
        elif weight_type == 'add_multiplicative':
            active_type = Add_Multive_Weight

        Linear_Active_ = partial(Linear_Active, active_type = active_type, n_context = n_context, gain_w = gain_w, gain_b = gain_b, passive = passive)

        super().__init__(n_arch, n_context, nonlin, loss_fnc, device, FC_module = Linear_Active_)


    def forward(self, x, ctx_list):
        ctx = torch.cat(ctx_list, dim=1) 

        for i, module in enumerate(self.module_list):
            x = module(x, ctx)
            if i < len(self.module_list) - 1:  
                x = self.nonlin(x)
        return x



class Model_Additive(Model_Active):
    def __init__(self, n_arch, n_context, gain_w = 1, gain_b = 1, nonlin=nn.ReLU(), passive=True, device=None): 
        super().__init__(n_arch, n_context, weight_type = 'additive', gain_w = gain_w, gain_b = gain_b, nonlin=nonlin, passive=passive, device=device)

class Model_Multiplicative(Model_Active):
    def __init__(self, n_arch, n_context, gain_w = 1, gain_b = 1, nonlin=nn.ReLU(), passive=True, device=None): 
        super().__init__(n_arch, n_context, weight_type = 'multiplicative', gain_w = gain_w, gain_b = gain_b, nonlin=nonlin, passive=passive, device=device)

class Model_Add_Multiplicative(Model_Active):
    def __init__(self, n_arch, n_context, gain_w = 1, gain_b = 1, nonlin=nn.ReLU(), passive=True, device=None): 
        super().__init__(n_arch, n_context, weight_type = 'add_multiplicative', gain_w = gain_w, gain_b = gain_b, nonlin=nonlin, passive=passive, device=device)

######################################

class Linear_Active(nn.Module):
    def __init__(self, n_input, n_output, active_type, n_context, gain_w, gain_b, passive): 
        super().__init__()
            
        self.active_type = active_type(n_input, n_output, n_context, gain_w, gain_b) #, passive)
        
    def forward(self, x, context):
        weight, bias = self.active_type(context)
        batch_size = context.shape[0]
        
        if batch_size > 1:
            x = torch.einsum('bs,bas->ba', x, weight) + bias
        else:
            x = F.linear(x, weight, bias)
        
        return x


######################################




class Active_Weight(nn.Module):
    def __init__(self): 
        super().__init__()

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



class Multive_Weight(Active_Weight):
    def __init__(self, n_input, n_output, n_context, gain_active=1, gain_passive=1): # passive=True):  
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        w0, b0 = self.initialize(n_input, n_output, gain_passive)
        self.w0        = Parameter(w0)
        self.b0        = Parameter(b0)

        # self.w_passive, self.b_passive = 0, 0
        # if passive:
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

            
    def forward(self, context):
        batch_size = context.shape[0]
        o_w = F.linear(context, self.w_active, self.w_passive).view(batch_size, self.n_output, self.n_input)
        o_b = F.linear(context, self.b_active, self.b_passive).view(batch_size, self.n_output)

        if batch_size == 1:
            o_w.squeeze_(dim=0)
            o_b.squeeze_(dim=0)

        weight = self.w0 * F.sigmoid(o_w)
        bias   = self.b0 * F.sigmoid(o_b)
        return weight, bias

    
class Additive_Weight(Active_Weight):
    def __init__(self, n_input, n_output, n_context, gain_active=1, gain_passive=1): #, passive=True): 
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        # self.w_passive, self.b_passive = 0, 0

        w0, b0 = self.initialize(n_input, n_output, gain_passive)
        self.w0        = Parameter(w0)
        self.b0        = Parameter(b0)

        if n_context > 0:
            w_all, b_all = [], []
            for i in range(n_context):
                w, b = self.initialize(n_input, n_output, gain_active)
                w_all.append(w)
                b_all.append(b)
            self.w_active = Parameter(torch.stack(w_all, dim=2).view(-1, n_context))
            self.b_active = Parameter(torch.stack(b_all, dim=1))

            
    def forward(self, context):
        batch_size = context.shape[0]
        d_w = F.linear(context, self.w_active, torch.zeros(1)).view(batch_size, self.n_output, self.n_input)
        d_b = F.linear(context, self.b_active, torch.zeros(1)).view(batch_size, self.n_output)
        if batch_size == 1:
            d_w.squeeze_(dim=0)
            d_b.squeeze_(dim=0)
        
        weight = self.w0 + d_w
        bias   = self.b0 + d_b
        return weight, bias



class Add_Multive_Weight(Active_Weight):
    def __init__(self, n_input, n_output, n_context, gain_active=1, gain_passive=1): # passive=True):  
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        w0, b0 = self.initialize(n_input, n_output, gain_passive)
        self.w0        = Parameter(w0)
        self.b0        = Parameter(b0)

        w, b = self.initialize(n_input, n_output, gain_passive)
        self.w_passive = Parameter(w.view(-1))
        self.b_passive = Parameter(b)

        w_all, b_all = [], []
        for i in range(n_context):
            w, b = self.initialize(n_input, n_output, gain_active)
            w_all.append(w)
            b_all.append(b)
        self.w_additive = Parameter(torch.stack(w_all, dim=2).view(-1, n_context))
        self.b_additive = Parameter(torch.stack(b_all, dim=1))

        w_all, b_all = [], []
        for i in range(n_context):
            w, b = self.initialize(n_input, n_output, gain_active)
            w_all.append(w)
            b_all.append(b)
        self.w_multiplicative = Parameter(torch.stack(w_all, dim=2).view(-1, n_context))
        self.b_multiplicative = Parameter(torch.stack(b_all, dim=1))
            
    def forward(self, context):
        batch_size = context.shape[0]
        o_w = F.linear(context, self.w_multiplicative, self.w_passive).view(batch_size, self.n_output, self.n_input)
        o_b = F.linear(context, self.b_multiplicative, self.b_passive).view(batch_size, self.n_output)
        d_w = F.linear(context, self.w_additive, torch.zeros(1)).view(batch_size, self.n_output, self.n_input)
        d_b = F.linear(context, self.b_additive, torch.zeros(1)).view(batch_size, self.n_output)

        if batch_size == 1:
            o_w.squeeze_(dim=0)
            o_b.squeeze_(dim=0)
            d_w.squeeze_(dim=0)
            d_b.squeeze_(dim=0)

        weight = self.w0 * F.sigmoid(o_w) + d_w
        bias   = self.b0 * F.sigmoid(o_b) + d_b
        return weight, bias

######################################

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

class HSML(nn.Module):
    def __init__(self, input_dim, n_hidden, tree_hidden_dim, cluster_layer_0, cluster_layer_1):

        # Initialize variables (hidden dim, input dim, output dim, )
        self.n_context = n_hidden # output dimension of task embedding
        self.tree_hidden_dim = tree_hidden_dim

        # Initialize parameters for HSML
        self.cluster_layer_0 = cluster_layer_0
        self.cluster_layer_1 = cluster_layer_1
        self._initialize_all()

        # Initialize parameters for part (b)
        self._construct_cluster()

        self.encoder = Encoder_Core(input_dim, n_hidden)

    def _construct_cluster(self):
        self.cluster = []
        self.weight_0 = []
        self.bias_0 = []
        for i in range(self.cluster_layer_0):
            self.cluster.append(torch.nn.Parameter(torch.empty(1,self.n_context)))
            self.weight_0.append(torch.nn.Parameter(torch.empty(self.n_context,self.tree_hidden_dim)))
            self.bias_0.append(torch.nn.Parameter(torch.empty(1,self.tree_hidden_dim)))

        self.cluster_weight_1, self.cluster_bias_1 = [], []
        self.weight_1, self.bias_1 = [], []
        for i in range(self.cluster_layer_1):
            self.cluster_weight_1.append(torch.nn.Parameter(torch.empty(self.tree_hidden_dim,1)))
            self.cluster_bias_1.append(torch.nn.Parameter(torch.empty(1,1)))
            self.weight_1.append(torch.nn.Parameter(torch.empty(self.tree_hidden_dim,self.tree_hidden_dim)))
            self.bias_1.append(torch.nn.Parameter(torch.empty(1,self.tree_hidden_dim)))

        self.weight_2 = torch.nn.Parameter(torch.empty(self.tree_hidden_dim,self.tree_hidden_dim))
        self.bias_2 = torch.nn.Parameter(torch.empty(1,self.tree_hidden_dim))

    def _clustering(self, h): # Part (b)
        sigma = 2.0

        for idx in range(self.cluster_layer_0):
            if idx == 0:
                all_value = torch.exp(-torch.sum((h - self.cluster[idx])**2) / (2.0 * sigma))
            else:
                all_value += torch.exp(-torch.sum((h - self.cluster[idx])**2) / (2.0 * sigma))

        h_0 = []
        for idx in range(self.cluster_layer_0):
            assignment_idx = torch.exp(-torch.sum((h - self.cluster[idx])**2) / (2.0 * sigma)) / all_value
            value_u = torch.tanh(torch.mm(h, self.weight_0[idx]) + self.bias_0[idx])
            value_c = assignment_idx * value_u
            h_0.append(value_c)

        h_1 = []
        for idx in range(self.cluster_layer_0):
            p_0 = []
            for idx_layer_1 in range(self.cluster_layer_1):
                p_0.append(torch.mm(h_0[idx], self.cluster_weight_1[idx_layer_1]) + self.cluster_bias_1[idx_layer_1])
            p_0 = F.softmax(torch.cat(p_0, dim=0), dim=0)

            h_1_temp = []
            for idx_layer_1 in range(self.cluster_layer_1):
                h_1_temp.append(p_0[idx_layer_1] * torch.tanh(torch.mm(h_0[idx], self.weight_1[idx_layer_1]) + self.bias_1[idx_layer_1]))
            h_1.append(torch.cat(h_1_temp, dim=0))

        h_1 = torch.stack(h_1, dim=0)
        h_1 = h_1.permute(1, 0, 2)
        h_1 = torch.sum(h_1, dim=1)

        g = []
        for idx in range(self.cluster_layer_1):
            g.append(torch.tanh(torch.mm(h_1[idx].unsqueeze(0), self.weight_2) + self.bias_2))

        g = torch.sum(torch.cat(g, axis=0), dim=0)

        return g

    def _initialize_all(self):
        self._initialize(self.weight_0)
        self._initialize(self.bias_0)

        self._initialize(self.cluster_weight_1)
        self._initialize(self.weight_1)
        self._initialize(self.cluster_bias_1)
        self._initialize(self.bias_1)

        self.weight_2.data = torch.initialize(self.weight_2)
        self.bias_2.data = torch.initialize(self.bias_2)

        self._initialize(self.cluster)

    def _initialize(param_list):
        for par in param_list:
            par.data = torch.initialize(par)

    def forward(self, input):
        h = self.encoder.forward(input, progressive = None):
        g = self._clustering(h)
        g_embed = torch.cat((h, g), dim=1)
        return g_embed
