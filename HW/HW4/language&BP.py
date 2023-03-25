import os
import glob
import random
from collections import Counter
import numpy as np
import re

from google.colab import drive

drive.mount('/content/drive')

def natural_sort_key(s):
    """A natural sort key for file names."""
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', s)]




data_dir = '/content/drive/MyDrive/languageID/languageID'
file_paths = sorted(glob.glob(os.path.join(data_dir, '*.txt')), key=natural_sort_key)


data = []
labels = []

for file_path in file_paths:
    with open(file_path, 'r') as f:
        document = f.read()
        label = file_path.split('/')[-1][0]  # extract first character of filename as label
        data.append(document)
        labels.append(label)

alpha = 0.5
english_data = [d for d, l in zip(data, labels) if l == 'e']
english_train_data = english_data[:10]

# Count the occurrences of each character in English training documents
char_counts = np.zeros((27,))
for document in english_train_data:
    for c in document:
        if c == ' ':  # space character
            char_counts[26] += 1
        elif 'a' <= c <= 'z':  # alphabetic character
            char_counts[ord(c) - ord('a')] += 1

# Calculate the class conditional probabilities for English
theta_e = (char_counts + alpha) / (np.sum(char_counts) + 27 * alpha)

print(theta_e)

japanese_data = [d for d, l in zip(data, labels) if l == 'j']
japanese_train_data = japanese_data[:10]

# Count the occurrences of each character in English training documents
char_counts = np.zeros((27,))
for document in japanese_train_data:
    for c in document:
        if c == ' ':  # space character
            char_counts[26] += 1
        elif 'a' <= c <= 'z':  # alphabetic character
            char_counts[ord(c) - ord('a')] += 1

# Calculate the class conditional probabilities for English
theta_j = (char_counts + alpha) / (np.sum(char_counts) + 27 * alpha)

print(theta_j)

spanish_data = [d for d, l in zip(data, labels) if l == 's']
spanish_train_data = spanish_data[:10]

# Count the occurrences of each character in English training documents
char_counts = np.zeros((27,))
for document in spanish_train_data:
    for c in document:
        if c == ' ':  # space character
            char_counts[26] += 1
        elif 'a' <= c <= 'z':  # alphabetic character
            char_counts[ord(c) - ord('a')] += 1

# Calculate the class conditional probabilities for English
theta_s = (char_counts + alpha) / (np.sum(char_counts) + 27 * alpha)

print(theta_s)

test_file = '/content/drive/MyDrive/languageID/languageID/e10.txt'

with open(test_file, 'r') as f:
    test_doc = f.read()

# Count the occurrences of each character in the test document
test_counts = np.zeros((27,))
for c in test_doc:
    if c == ' ':  # space character
        test_counts[26] += 1
    elif 'a' <= c <= 'z':  # alphabetic character
        test_counts[ord(c) - ord('a')] += 1

print(test_counts)

# Compute log p(x|y=e)
log_p_x_given_e = np.sum(test_counts * np.log(theta_e))
print("log p(x|y=e) = {:.2f}".format(log_p_x_given_e))

# Compute log p(x|y=j)
log_p_x_given_j = np.sum(test_counts * np.log(theta_j))
print("log p(x|y=j) = {:.2f}".format(log_p_x_given_j))

# Compute log p(x|y=s)
log_p_x_given_s = np.sum(test_counts * np.log(theta_s))
print("log p(x|y=s) = {:.2f}".format(log_p_x_given_s))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Define the hyperparameters
batch_size = 32
learning_rate = 0.1
num_epochs = 100

# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self, d, d1=300, d2=200, k=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, k)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x
    

# Define the network and optimizer
net = Net(784, 300, 200, 10)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

loss_track = []
train_track = []

# Train the network
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.view(-1, 784))
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            outputs = net(inputs.view(-1, 784))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    loss_track.append(running_loss / len(trainloader))
    train_track.append(correct / total)
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
    print('[%d] training accuracy: %.2f %%' % (epoch + 1, (100 * correct / total)))



# Test the network
correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs.view(-1, 784))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test accuracy (random): %.2f %%' % (100 * correct / total))
print('Test error (random): %.2f %%' % (100-(100 * correct / total)))

plt.figure()
plt.plot(loss_track)
plt.title('Learning Curve (initialize zero)')
plt.show()

plt.figure()
plt.plot(train_track)
plt.title('training accuracy (pytorch, random)')
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Define the hyperparameters
batch_size = 64
learning_rate = 0.1
num_epochs = 100

# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self, d, d1=300, d2=200, k=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, k)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x
    

# Define the network and optimizer
net = Net(784, 300, 200, 10)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

loss_track = []
train_track = []

# Train the network
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.view(-1, 784))
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            outputs = net(inputs.view(-1, 784))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    loss_track.append(running_loss / len(trainloader))
    train_track.append(correct / total)
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
    print('[%d] training accuracy: %.2f %%' % (epoch + 1, (100 * correct / total)))


plt.figure()
plt.plot(loss_track)
plt.title('learning curve (pytorch, random)')
plt.show()

plt.figure()
plt.plot(train_track)
plt.title('training accuracy (pytorch, random)')
plt.show()
