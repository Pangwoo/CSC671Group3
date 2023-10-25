#Main body

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
testing_data = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())

# Set the batch size for the loaders
batch_size = 100

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(list(len(training_data)))
test_sampler = SubsetRandomSampler(list(len(testing_data)))
# For further validating of training data "valid_sampler = SubsetRandomSampler(valid_idx)

#define the transformation to normalize value to [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
    sampler=train_sampler, transform= transform)
#valid_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, transform= transform)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

