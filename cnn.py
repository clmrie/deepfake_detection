import torch
import torch.nn as nn
from torch.nn import Module # pytorch using classes to build nn
from torch.nn import Conv2d # pytorch implementation conv layer
from torch.nn import Linear # pytorch fully connected layer
from torch.nn import MaxPool2d # 2d max-pooling reduce dimensions on input volume
from torch.nn import ReLU # activation function
from torch.nn import LogSoftmax # log of softmax function (softmax classifier returns predicted prob of each class)
from torch import flatten # flattens output of multi-dimensional, from conv/pooling layers to apply FC layers


# Model architecture
# -*- coding:utf-8 -*-

IMGWIDTH = 256

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = None
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)


class Meso1(Classifier):
    """
    Feature extraction + Classification
    """
    def __init__(self, learning_rate=0.001, dl_rate=1):
        super(Meso1, self).__init__()
        self.model = self.init_model(dl_rate)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def init_model(self, dl_rate):
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, dilation=dl_rate, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=(8, 8)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(4 * (IMGWIDTH // 8) * (IMGWIDTH // 8), 1),
            nn.Sigmoid()
        )
        return model


class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        super(Meso4, self).__init__()
        self.model = self.init_model()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def init_model(self): 
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * (IMGWIDTH // 64) * (IMGWIDTH // 64), 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        return model


in_channels = 256
out_channels = 2

class MesoInception4(Classifier):
    def __init__(self, learning_rate=0.001):
        super(MesoInception4, self).__init__()
        self.model = self.init_model()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def InceptionLayer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return layer
    
    def init_model(self):
        model = nn.Sequential(
            self.InceptionLayer(3, 11),
            nn.BatchNorm2d(11),
            nn.MaxPool2d(kernel_size=2),
            self.InceptionLayer(11, 11),
            nn.BatchNorm2d(11),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(11, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * (IMGWIDTH // 64) * (IMGWIDTH // 64), 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        return model


class MesoInception4_v2(nn.Module):
	"""
	Pytorch Implemention of MesoInception4
	Author: Honggu Liu
	Date: July 7, 2019
	"""
	def __init__(self, num_classes=2):
		super(MesoInception4, self).__init__()
		self.num_classes = num_classes
		#InceptionLayer1
		self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
		self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
		self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption1_bn = nn.BatchNorm2d(11)


		#InceptionLayer2
		self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption2_bn = nn.BatchNorm2d(12)

		#Normal Layer
		self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.bn1 = nn.BatchNorm2d(16)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.fc2 = nn.Linear(16, num_classes)


	#InceptionLayer
	def InceptionLayer1(self, input):
		x1 = self.Incption1_conv1(input)
		x2 = self.Incption1_conv2_1(input)
		x2 = self.Incption1_conv2_2(x2)
		x3 = self.Incption1_conv3_1(input)
		x3 = self.Incption1_conv3_2(x3)
		x4 = self.Incption1_conv4_1(input)
		x4 = self.Incption1_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption1_bn(y)
		y = self.maxpooling1(y)

		return y

	def InceptionLayer2(self, input):
		x1 = self.Incption2_conv1(input)
		x2 = self.Incption2_conv2_1(input)
		x2 = self.Incption2_conv2_2(x2)
		x3 = self.Incption2_conv3_1(input)
		x3 = self.Incption2_conv3_2(x3)
		x4 = self.Incption2_conv4_1(input)
		x4 = self.Incption2_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption2_bn(y)
		y = self.maxpooling1(y)

		return y

	def forward(self, input):
		x = self.InceptionLayer1(input) #(Batch, 11, 128, 128)
		x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

		x = self.conv1(x) #(Batch, 16, 64 ,64)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(Batch, 16, 32, 32)

		x = self.conv2(x) #(Batch, 16, 32, 32)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling2(x) #(Batch, 16, 8, 8)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x
