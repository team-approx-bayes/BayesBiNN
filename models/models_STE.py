import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torch.autograd import Function


class BinarizeF(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
binarize = BinarizeF.apply



class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output


class BinaryLinear(nn.Linear):

    def forward(self, input):
        # binary_weight = binarize(self.weight)
        #if input.size(1) != 784:
        #    input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


class BinConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        #if input.size(1) != 3:
        #    input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out



class MLPBinaryConnect_STE(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.15, eps=1e-4,drop_prob=0,batch_affine=True):
        super(MLPBinaryConnect_STE, self).__init__()
        self.in_features = in_features

        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)


        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = BinaryLinear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = BinaryLinear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = BinaryLinear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc4 = BinaryLinear(num_units, out_features, bias=False)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)


        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)


        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)


        x = self.fc4(x)
        x = self.bn4(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),



class VGGBinaryConnect_STE(nn.Module):
    """VGG-like net used for Cifar10.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
       We wirte separately for BCVI optimizer
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=True):
        super(VGGBinaryConnect_STE, self).__init__()
        self.in_features = in_features
        self.conv1 = BinConv2d(in_features, 128, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv2 = BinConv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv3 = BinConv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv4 = BinConv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv5 = BinConv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv6 = BinConv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)


        self.fc1 = BinaryLinear(512 * 4 * 4, 1024, bias=False)
        self.bn7 = nn.BatchNorm1d(1024,affine=batch_affine)

        self.fc2 = BinaryLinear(1024, 1024, bias=False)
        self.bn8 = nn.BatchNorm1d(1024,affine=batch_affine)


        self.fc3 = BinaryLinear(1024, out_features, bias=False)
        self.bn9 = nn.BatchNorm1d(out_features,affine=batch_affine)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(x))

        x = F.relu(self.bn3(self.conv3(x)))


        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn4(x))

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn6(x))

        x = x.view(-1, 512 * 4 * 4)

        x = self.fc1(x)
        x = F.relu(self.bn7(x))

        x = self.fc2(x)
        x = F.relu(self.bn8(x))

        x = self.fc3(x)
        x = self.bn9(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),
