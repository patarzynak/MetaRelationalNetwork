import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    #print(current_dict.keys(), output_dict.keys())
    return output_dict

def cvt_coord(i, d):  #rewrite with greater generality
            return [(i/d-2)/2., (i%d-2)/2.]


def pair(x, qst):

    mb = x.size()[0]
    n_channels = x.size()[1]
    d = x.size()[3]

    #coord_tensor = torch.FloatTensor(mb, d*d, 2, requires_grad=True)
    coord_tensor = torch.ones(mb, d*d, 2, requires_grad=True)
    np_coord_tensor = np.zeros((mb, d*d, 2))
    for i in range(d*d):
        np_coord_tensor[:, i,:] = np.array( cvt_coord(i, d) )
    coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
    #coord_tensor = coord_tensor.new_tensor(np_coord_tensor, requires_grad=True)

    x_flat = x.view(mb, n_channels,d*d).permute(0,2,1)   #5 * 3*3 * 24

    x_flat = torch.cat([x_flat, coord_tensor], 2)
    
    qst = qst.repeat(mb, 1)
    qst = torch.unsqueeze(qst, 1)
    qst = qst.repeat(1,d*d,1)
    qst = torch.unsqueeze(qst, 2)

    # cast all pairs against each other
    x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
    x_i = x_i.repeat(1,d*d,1,1) # (64x25x25x26+11)
    x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)


    x_j = torch.cat([x_j,qst],3)
    x_j = x_j.repeat(1,1,d*d,1) # (64x25x25x26+11)


    x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)

    return x_full

class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1, layer_num =1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        ##delete line below and from the rn too etc
        self.layer_num = layer_num
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            #print("No inner loop params")
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        #print("PARAMS", params)
        # if self.layer_num == 1:
        #     print(params['weight'][0][0][0][2])
        #print("OUT", out)
        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass
            #print('no inner loop params', self)

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class ConvInputModel(nn.Module):
    def __init__(self, device, args):
        super(ConvInputModel, self).__init__()

        self.layer_dict = nn.ModuleDict()
        
        self.layer_dict['conv1'] = MetaConv2dLayer(3, 24, 3, stride=2, padding=1, use_bias=True, layer_num=1)
        self.layer_dict['batchNorm1'] = MetaBatchNormLayer(24, device=device, args=args)
        self.layer_dict['conv2'] = MetaConv2dLayer(24, 24, 3, stride=2, padding=1, use_bias=True, layer_num=2)
        self.layer_dict['batchNorm2'] = MetaBatchNormLayer(24, device=device, args=args)
        self.layer_dict['conv3'] = MetaConv2dLayer(24, 24, 3, stride=1, padding=1, use_bias=True, layer_num=3)
        self.layer_dict['batchNorm3'] = MetaBatchNormLayer(24, device=device, args=args)
        self.layer_dict['conv4'] = MetaConv2dLayer(24, 24, 3, stride=2, padding=1, use_bias=True, layer_num=4)
        self.layer_dict['batchNorm4'] = MetaBatchNormLayer(24, device=device, args=args)

        
    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """convolution"""
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        print(x.size())
        x = self.layer_dict['conv1'](x, params=param_dict['conv1'])
        x = F.leaky_relu(x)
        x = self.layer_dict['batchNorm1'](x, num_step, params=param_dict['batchNorm1'], training=training, backup_running_statistics=backup_running_statistics)
        print(x.size())

        x = self.layer_dict['conv2'](x, params=param_dict['conv2'])
        x = F.leaky_relu(x)
        x = self.layer_dict['batchNorm2'](x, num_step, params=param_dict['batchNorm2'], training=training, backup_running_statistics=backup_running_statistics)
        print(x.size())
        x = self.layer_dict['conv3'](x, params=param_dict['conv3'])
        x = F.leaky_relu(x)
        x = self.layer_dict['batchNorm3'](x, num_step, params=param_dict['batchNorm3'], training=training, backup_running_statistics=backup_running_statistics)
        print(x.size())
        x = self.layer_dict['conv4'](x, params=param_dict['conv4'])
        x = F.leaky_relu(x)
        x = self.layer_dict['batchNorm4'](x, num_step, params=param_dict['batchNorm4'], training=training, backup_running_statistics=backup_running_statistics)
        #print(x)
        print(x.size())
        return x

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        self.layer_dict['batchNorm1'].restore_backup_stats()
        self.layer_dict['batchNorm2'].restore_backup_stats()
        self.layer_dict['batchNorm3'].restore_backup_stats()
        self.layer_dict['batchNorm4'].restore_backup_stats()

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.layer_dict = nn.ModuleDict()

        self.layer_dict['fc2'] = MetaLinearLayer(256, 256)
        self.layer_dict['fc3'] = MetaLinearLayer(256, 3)
        self.layer_dict['dropout'] = nn.Dropout()

    def forward(self, x, params=None):
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None
        
        x = self.layer_dict['fc2'](x, params=param_dict['fc2'])
        x = F.leaky_relu(x)
        x = self.layer_dict['dropout'](x)
        x = self.layer_dict['fc3'](x, params=param_dict['fc3'])
        #return F.log_softmax(x, dim=1)
        
        #return F.softmax(x)
        return x


class RN(nn.Module):
    def __init__(self, device, args):
        super(RN, self).__init__()
        
        self.layer_dict = nn.ModuleDict()

        self.args = args
        self.device = device

        self.layer_dict['conv'] = ConvInputModel(device=device, args=args)
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.layer_dict['g_fc1'] = MetaLinearLayer((24+2)*2+33, 256)

        self.layer_dict['g_fc2'] = MetaLinearLayer(256, 256)
        self.layer_dict['g_fc3'] = MetaLinearLayer(256, 256)
        self.layer_dict['g_fc4'] = MetaLinearLayer(256, 256)

        self.layer_dict['f_fc1'] = MetaLinearLayer(256, 256)

        self.layer_dict['fcout'] = FCOutputModel()


    def forward(self, x, question, num_step, params=None, training=False, backup_running_statistics=False):
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None
        
        x = self.layer_dict['conv'](
            x,
            num_step,
            params=param_dict['conv'],
            training=training,
            backup_running_statistics=backup_running_statistics

        ) ## x = (64 x 24 x 5 x 5)
        mb = x.size()[0]
        d = x.size()[2]
        #print("hhhhhhhhhhhhhhhhhhhhhhhh", x.size())

        #x_full = pair(x, question)
        #print(x_full)
        #mb = x.size()[0]
        n_channels = x.size()[1]
        #d = x.size()[3]

        #coord_tensor = torch.FloatTensor(mb, d*d, 2, requires_grad=True)
        coord_tensor = torch.ones(mb, d*d, 2, requires_grad=True, device=self.device)
        np_coord_tensor = np.zeros((mb, d*d, 2))
        for i in range(d*d):
            np_coord_tensor[:, i,:] = np.array( cvt_coord(i, d) )
        coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        #coord_tensor = coord_tensor.new_tensor(np_coord_tensor, requires_grad=True)

        x_flat = x.view(mb, n_channels,d*d).permute(0,2,1)

        x_flat = torch.cat([x_flat, coord_tensor], 2)
        
        question = question.repeat(mb, 1)
        question = torch.unsqueeze(question, 1)
        question = question.repeat(1,d*d,1)
        question = torch.unsqueeze(question, 2)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,d*d,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)


        x_j = torch.cat([x_j,question],3)
        x_j = x_j.repeat(1,1,d*d,1) # (64x25x25x26+11)


        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)




  
        # (24+2)*2+33 = 85
        x_ = x_full.view(mb*d*d*d*d,85)
        x_ = self.layer_dict['g_fc1'](x_, params=param_dict['g_fc1'])
        x_ = F.leaky_relu(x_)
        x_ = self.layer_dict['g_fc2'](x_, params=param_dict['g_fc2'])
        x_ = F.leaky_relu(x_)
        x_ = self.layer_dict['g_fc3'](x_, params=param_dict['g_fc3'])
        x_ = F.leaky_relu(x_)
        x_ = self.layer_dict['g_fc4'](x_, params=param_dict['g_fc4'])
        x_ = F.leaky_relu(x_)

            
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.layer_dict['f_fc1'](x_g, params=param_dict['f_fc1'])
        x_f = F.leaky_relu(x_f)
        

        preds = self.layer_dict['fcout'](x_f, params=param_dict['fcout'])

        #print("preds", preds)
        return preds

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            #print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            #print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        self.layer_dict['conv'].restore_backup_stats() 


class CNN_MLP(nn.Module):
    def __init__(self, device, args):
        super(CNN_MLP, self).__init__()

        self.args = args
        self.device = device

        self.layer_dict = nn.ModuleDict()

        self.layer_dict['conv']  = ConvInputModel(device=device, args=args)
        self.layer_dict['fc1']   = MetaLinearLayer(6*6*24 + 33, 256)  # question concatenated to all
        self.layer_dict['fcout'] = FCOutputModel()

        #print([ a for a in self.parameters() ] )
  
    def forward(self, x, question, num_step, params=None, training=False, backup_running_statistics=False):
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        x = self.layer_dict['conv'](
            x,
            num_step,
            params=param_dict['conv'],
            training=training,
            backup_running_statistics=backup_running_statistics
        ) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)

        question = question.repeat(x.size(0), 1)
        x_ = torch.cat((x, question), 1)  # Concat question
        
        x_ = self.layer_dict['fc1'](x_, params=param_dict['fc1'])
        x_ = F.leaky_relu(x_)
        
        preds = self.layer_dict['fcout'](x_, params=param_dict['fcout'])

        return preds

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        self.layer_dict['conv'].restore_backup_stats() 

