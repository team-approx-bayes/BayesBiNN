import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBinaryConnect(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, num_units=2048, momentum=0.15, eps=1e-4,drop_prob=0,batch_affine=False):
        super(MLPBinaryConnect, self).__init__()
        self.in_features = in_features

        self.dropout1 = nn.Dropout(p=drop_prob)

        self.dropout2 = nn.Dropout(p=drop_prob)


        self.dropout3 = nn.Dropout(p=drop_prob)

        self.dropout4 = nn.Dropout(p=drop_prob)


        self.fc1 = nn.Linear(in_features, num_units, bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = nn.Linear(num_units, num_units, bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = nn.Linear(num_units, num_units, bias=False)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc4 = nn.Linear(num_units, out_features, bias=False)
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

class MLP_CL_h100(nn.Module):
    """Multi-Layer Perceptron used for MNIST. No convolution layers.
       This model is the MLP architecture for continual learning
    """
    def __init__(self, in_features, out_features, num_units=100, momentum=0.15, eps=1e-4,batch_affine=True):
        super(MLP_CL_h100, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, num_units,bias=False)
        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc2 = nn.Linear(num_units, num_units,bias=False)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum,affine=batch_affine)

        self.fc3 = nn.Linear(num_units, out_features,bias=False)
        self.bn3 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum,affine=batch_affine)


    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)

        return x  # if used NLL loss, the output should be changed to F.log_softmax(x,dim=1),



class VGGBinaryConnect(nn.Module):
    """VGG-like net used for Cifar10.
       This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False):
        super(VGGBinaryConnect, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, 128, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=eps, momentum=momentum,affine=batch_affine)


        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(512, eps=eps, momentum=momentum,affine=batch_affine)


        self.fc1 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.bn7 = nn.BatchNorm1d(1024,affine=batch_affine)

        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn8 = nn.BatchNorm1d(1024,affine=batch_affine)


        self.fc3 = nn.Linear(1024, out_features, bias=False)
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





    







