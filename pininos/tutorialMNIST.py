# imports
import torch 
import torch.nn as nn
import torch.optim as optim #optimization algorithms 
import torch.nn.functional as F # functions with no parameters relu, tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms # transformations in the dataset
from tqdm import tqdm

# create fully connected network
class NN(nn.Module):
    def __init__(self,  input_size, num_classes): #28 x 28 
        super(NN, self).__init__() #cause the initialization of the parent 
        self.fc1 = nn.Linear(input_size, 50) #hideen layer of 50 nodes
        self.fc2 = nn.Linear(50, input_size)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
model = NN(784,10) #10 is the number of digits
x = torch.randn(64,784) # number of exmaples we will run simustal, n of images
print(model(x).shape)
"""

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

#load data
train_dataset = datasets.MNIST(root = "dataset/", train= True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = "dataset/", train= False, transform = transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)

# initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        #print(data.shape)
        data = data.reshape(data.shape[0], -1) #correct shape

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step 
        optimizer.step()


# check accuracy on training and test to see how good our model
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0 
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}/ {num_samples} with accuract {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    return num_correct / num_samples

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)