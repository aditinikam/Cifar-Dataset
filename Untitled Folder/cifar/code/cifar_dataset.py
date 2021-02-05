import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from PIL import Image
class customDatasetClass():

    def __init__(self,img,label):

        # self.path = path
        self.allImagePaths = []
        self.allTargets = []
        # self.targetToClass = sorted(os.listdir(self.path))

        # for targetNo, targetI in enumerate(self.targetToClass):
        #     for imageI in sorted(os.listdir(self.path + '/' + targetI)):
        #         self.allImagePaths.append(self.path + '/' + targetI + '/' + imageI)
        #         self.allTargets.append(targetNo)

        self.allImagePaths= img
        self.allTargets=label

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((256, 256)),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item):

        # image = Image.open(self.allImagePaths[item]).convert('RGB')
        target = self.allTargets[item]
        image = self.allImagePaths[item]

        return image, target

    def __len__(self):

        return len(self.allImagePaths)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(3072, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        # x - shape = [64, 784]
        # Layer 1
        l1 = self.fc1(x)  # l1 - shape = [64, 100]

        # Activation 1
        l1_a1 = torch.sigmoid(l1)  # l1_a1 - shape = [64, 100]

        # Layer 2
        l2 = self.fc2(l1_a1)  # l2 - shape = [64, 10]

        # Activation 2
        l2_a2 = torch.sigmoid(l2)  # l2_a2 - shape = [64, 10]
        
        return l2_a2

def train(model, use_cuda, train_loader, optimizer, epoch):

    model.train()  # Tell the model to prepare for training
    
    for batch_idx, (data, target) in enumerate(train_loader):  # Get the batch

        # Converting the target to one-hot-encoding from categorical encoding
        # Converting the data to [batch_size, 784] from [batch_size, 1, 28, 28]

        y_onehot = torch.zeros([target.shape[0], 10])  # Zero vector of shape [64, 10]
        y_onehot[range(target.shape[0]), target.long()] = 1
        

        data = data.view([data.shape[0], 3072])

        if use_cuda:
            data, y_onehot = data.cuda(), y_onehot.cuda()  # Sending the data to the GPU

        optimizer.zero_grad()  # Setting the cumulative gradients to 0
        output = model(data.float())  # Forward pass through the model
        loss = torch.mean((output - y_onehot)**2)  # Calculating the loss
        loss.backward()  # Calculating the gradients of the model. Note that the model has not yet been updated.
        optimizer.step()  # Updating the model parameters. Note that this does not remove the stored gradients!

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, use_cuda, test_loader):

    model.eval()  # Tell the model to prepare for testing or evaluation

    test_loss = 0
    correct = 0

    with torch.no_grad():  # Tell the model that gradients need not be calculated
        for data, target in test_loader:  # Get the batch

            # Converting the target to one-hot-encoding from categorical encoding
            # Converting the data to [batch_size, 784] from [batch_size, 1, 28, 28]

            y_onehot = torch.zeros([target.shape[0], 10])
            # y_onehot[range(target.shape[0]), target] = 1

            for idx, val in enumerate(target.shape):
                y_onehot[idx][val] = 1

            data = data.view([data.shape[0], 3072])

            if use_cuda:
                data, target, y_onehot = data.cuda(), target.cuda(), y_onehot.cuda()  # Sending the data to the GPU

            # argmax([0.1, 0.2, 0.9, 0.4]) => 2
            # output - shape = [1000, 10], argmax(dim=1) => [1000]
            output = model(data.float())  # Forward pass
            test_loss += torch.sum((output - y_onehot)**2)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the maximum output
            correct += pred.eq(target.view_as(pred)).sum().item()  # Get total number of correct samples

    test_loss /= len(test_loader.dataset)  # Accuracy = Total Correct / Total Samples

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def seed(seed_value):

    # This removes randomness, makes everything deterministic

    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():

    use_cuda = False  # Set it to False if you are using a CPU
    # Colab And Kaggle

    seed(0)

    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = np.load('trainImages.npy')
    trainLabel = np.load('trainLabels.npy')
    # trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    trainloader=DataLoader(
        customDatasetClass(trainset,trainLabel),
        batch_size=20,
        num_workers=2,
        shuffle=True
    )

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = Net()  # Get the model

    if use_cuda:
        model = model.cuda()  # Put the model weights on GPU

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Choose the optimizer and the set the learning rate
    testset = np.load('testImages.npy')
    testLabel=np.load('testLabels.npy')
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)
    testloader=DataLoader(
        customDatasetClass(testset,testLabel),
        batch_size=20,
        num_workers=2,
        shuffle=True
    )

    for epoch in range(1, 10 + 1):
        train(model, use_cuda, trainloader, optimizer, epoch)  # Train the network
        test(model, use_cuda, testloader)  # Test the network


    torch.save(model.state_dict(), "cifar.pt")

    model.load_state_dict(torch.load('cifar.pt'))


if __name__ == '__main__':
    main()
