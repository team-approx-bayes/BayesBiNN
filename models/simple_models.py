# This part is written by Roman Bachmann

from .models_STE import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# this is the network used for synthetic data

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func='tanh', output_var=False, bias=True, use_bn=False, learn_bn=True, only_last_bn=True):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_var = output_var
        self.use_bn = use_bn
        self.only_last_bn = only_last_bn
        bn_momentum = 0.15
        bn_eps = 1e-4

        if output_size is not None:
            self.output_size = output_size
        else :
            self.output_size = 1

        # Set activation function
        if act_func == 'relu':
            self.act = torch.relu
        elif act_func == 'tanh':
            self.act = torch.tanh
        elif act_func == 'sigmoid':
            self.act = torch.sigmoid

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size, bias=bias)
            if use_bn:
                self.output_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
            if output_var:
                self.output_layer_logvar = nn.Linear(self.input_size, self.output_size, bias=bias)
                if use_bn:
                    self.output_logvar_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            if use_bn and not only_last_bn:
                self.batch_norms = nn.ModuleList([nn.BatchNorm1d(in_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn) for in_size in hidden_sizes])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)
            if use_bn:
                self.output_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
            if output_var:
                self.output_layer_logvar = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)
                if use_bn:
                    self.output_logvar_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)

    def forward(self, x):
        x = x.view(-1,self.input_size)
        out = x
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            out = self.batch_norms[i](out) if self.use_bn and not self.only_last_bn else out
            out = self.act(out)
        z = self.output_layer(out)
        z = self.output_bn(z) if self.use_bn else z
        if self.output_var:
            z_logvar = self.output_layer_logvar(out)
            z_logvar = self.output_logvar_bn(z_logvar) if self.use_bn else z_logvar
            return z, z_logvar
        return z

    def predict(self, x):
        logits = self.forward(x)
        prob = torch.sigmoid(logits)
        return prob.reshape(-1).detach().numpy()

    def predict_multi(self, x):
        logits = self.forward(x)
        prob = torch.softmax(logits, dim=1)
        return prob.detach().numpy()


class MLP_STE(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func='tanh', output_var=False, bias=True, use_bn=False, learn_bn=True, only_last_bn=True):
        super(MLP_STE, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_var = output_var
        self.use_bn = use_bn
        self.only_last_bn = only_last_bn
        bn_momentum = 0.15
        bn_eps = 1e-4

        if output_size is not None:
            self.output_size = output_size
        else :
            self.output_size = 1

        # Set activation function
        if act_func == 'relu':
            self.act = torch.relu
        elif act_func == 'tanh':
            self.act = torch.tanh
        elif act_func == 'sigmoid':
            self.act = torch.sigmoid

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = BinaryLinear(self.input_size, self.output_size, bias=bias)
            if use_bn:
                self.output_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
            if output_var:
                self.output_layer_logvar = BinaryLinear(self.input_size, self.output_size, bias=bias)
                if use_bn:
                    self.output_logvar_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
        else:
            # Neural network
            self.hidden_layers = nn.ModuleList([BinaryLinear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            if use_bn and not only_last_bn:
                self.batch_norms = nn.ModuleList([nn.BatchNorm1d(in_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn) for in_size in hidden_sizes])
            self.output_layer = BinaryLinear(hidden_sizes[-1], self.output_size, bias=bias)
            if use_bn:
                self.output_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
            if output_var:
                self.output_layer_logvar = BinaryLinear(hidden_sizes[-1], self.output_size, bias=bias)
                if use_bn:
                    self.output_logvar_bn = nn.BatchNorm1d(self.output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)
    
    def forward(self, x):
        x = x.view(-1,self.input_size)
        out = x
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            out = self.batch_norms[i](out) if self.use_bn and not self.only_last_bn else out
            out = self.act(out)
        z = self.output_layer(out)
        z = self.output_bn(z) if self.use_bn else z
        if self.output_var:
            z_logvar = self.output_layer_logvar(out)
            z_logvar = self.output_logvar_bn(z_logvar) if self.use_bn else z_logvar
            return z, z_logvar
        return z

    def predict(self, x):
        logits = self.forward(x)
        prob = torch.sigmoid(logits)
        return prob.reshape(-1).detach().numpy()

    def predict_multi(self, x):
        logits = self.forward(x)
        prob = torch.softmax(logits, dim=1)
        return prob.detach().numpy()



class ConvNet(nn.Module):
    def __init__(self, n_classes=10):
        super(ConvNet, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        prob = F.softmax(logits, dim=1)
        return prob.detach().numpy()
