import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor, quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss,self).__init__()
        self.margin=1.0

    def squared_hinge_loss(self, input, target,num_classes=10):
            target = target.unsqueeze(1)
            target_onehot = torch.FloatTensor(target.size(0),num_classes).to(target.device)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1,target,1)
           # target = target.squeeze()
           # nb_digits = 10
            # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
           # y = torch.LongTensor(batch_size, 1).random_() % nb_digits
            # One hot encoding buffer that you create out of the loop and just keep reusing
           # y_onehot = torch.FloatTensor(batch_size, nb_digits)

            # In your for loop
           # y_onehot.zero_()
           # y_onehot.scatter_(1, y, 1)

           # target = 2 * y_onehot - 1
            #import pdb; pdb.set_trace()

            output=self.margin-input.mul(target_onehot)
            output[output.le(0)]=0
            output = torch.pow(output, 2) # squared ?? to make sure
            return output.mean()

    def forward(self, input, target):
        return self.squared_hinge_loss(input,target)


class SquaredHingeLoss100(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss100,self).__init__()
        self.margin=1.0

    def squared_hinge_loss(self, input, target,num_classes=100):
            target = target.unsqueeze(1)
            target_onehot = torch.FloatTensor(target.size(0),num_classes).to(target.device)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1,target,1)
           # target = target.squeeze()
           # nb_digits = 10
            # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
           # y = torch.LongTensor(batch_size, 1).random_() % nb_digits
            # One hot encoding buffer that you create out of the loop and just keep reusing
           # y_onehot = torch.FloatTensor(batch_size, nb_digits)

            # In your for loop
           # y_onehot.zero_()
           # y_onehot.scatter_(1, y, 1)

           # target = 2 * y_onehot - 1
            #import pdb; pdb.set_trace()

            output=self.margin-input.mul(target_onehot)
            output[output.le(0)]=0
            output = torch.pow(output, 2) # squared ?? to make sure
            return output.mean()

    def forward(self, input, target):
        return self.squared_hinge_loss(input,target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output
