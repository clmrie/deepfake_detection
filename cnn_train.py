# set the matplotlib backend so figures can be saved in the background
import argparse as args
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from cnn import Meso4
from sklearn.metrics import classification_report # classification report
from torch.utils.data import random_split # split dataset
from torch.utils.data import DataLoader # data laoding utility to build pipline to train cnn
from torchvision.transforms import ToTensor # converts input into tensors
from torch.optim import Adam # optimizer
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import os 
from torchvision.datasets import ImageFolder
import torch.optim as optim
from tqdm import tqdm

import torchvision.transforms as transforms


import augmented_images



class MesoInception4_v2(nn.Module):
	"""
	Pytorch Implemention of MesoInception4
	Author: Honggu Liu
	Date: July 7, 2019
	"""
	def __init__(self, num_classes=2):
		super(MesoInception4_v2, self).__init__()
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





# Traning loop
dataset_dir = "/raid/datasets/hackathon2024" # either
root_dir = os.path.expanduser("/kaggle/input/automathon-deepfake/dataset/experimental_dataset")

# define training hyperparameters
lr = 0.001
BATCH_SIZE = 64
EPOCHS = 2

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the augmented dataset
train_path = "augmented_images"

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_data = ImageFolder(train_path,
                         transform=train_transform)

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)


# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


model = MesoInception4_v2().to(device)
epochs = 1
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    #valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
    #                                             criterion)
    train_loss.append(train_epoch_loss)
    #valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    #valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    #print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    time.sleep(5)


'''
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

model = MesoInception4()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
train_model(model, train_data_loader, model.loss_fn, model.optimizer, num_epochs)




trainData = augmented_images(
    root="augmented_images",  
    train=True,
    download=False,  
    transform=ToTensor()
)

"""
validationData = resized_validation_images(
    root="resized_validation_images", 
    train=False,
    download=False,
    transform=ToTensor()
)
"""

testData = resized_test_images(
    root="resized_test_images", 
    train=False,
    download=False,
    transform=ToTensor()
)

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42)) # generator() randomize the splitting each time

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
	batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE
testSteps = len(testDataLoader.dataset) // BATCH_SIZE

# initialize the model
print("[INFO] initializing the LeNet model...")
model = MesoInception4(
	numChannels=1,
	classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()
 ------------------------------------------------------------- Start same training loop like in FNN: training & test & validation loop 
# loop over our epochs 
for e in range(0, EPOCHS):
	# set the model in training mode
	model.train()
	
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	totalTestLoss = 0
	
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	testCorrect = 0
	
	# loop over the training set
	for (x, y) in trainDataLoader:  # start a for loop over the DataLoader object
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))  # foward pass
		
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)
		
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()
		
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss  # collect the sum for loss of all batches
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
		
endTime = time.time()
duration = endTime - startTime
print("[INFO] training completed in {:.2f} seconds".format(duration))

# switch off autograd for evaluation (validation)
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # loop over the validation set
    for (x, y) in valDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))

        # make the predictions and calculate the validation loss
        pred = model(x)  # use data to make predicitons
        totalValLoss += lossFn(pred, y)

        # calculate the number of correct predictions
        valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
		

# calculate the average training and validation loss
avgTrainLoss = totalTrainLoss / trainSteps

# calculate the training and validation accuracy
trainCorrect = trainCorrect / len(trainDataLoader.dataset)

# update our training history
H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
H["train_acc"].append(trainCorrect)
H["val_loss"].append(avgValLoss.cpu().detach().numpy())
H["val_acc"].append(valCorrect)


# print the model training and validation information
print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
    avgTrainLoss, trainCorrect))



# --------------------------------------------------------- evaluate on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
    # Set the model in evaluation mode
    model.eval()

    # Initialize a list to store input-prediction tuples
    preds = []
    inputs = []

    # Loop over the test set
    for (x, y) in testDataLoader:  # x is input
        # Send the input to the device
        x = x.to(device)

        # Make the predictions
        pred = model(x)

        if pred >= 0.5:  # 1 is fake, 0 is true
            pred = 1
        else: 
            pred = 0

        # Append the input-prediction tuple to the list
        preds.append(pred)
        inputs.append(x)

df = pd.DataFrame(preds, columns=["id", "label"])
df.to_csv("submission.csv", index=False)	
		

'''