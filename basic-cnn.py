#this is just a basic model to solve the issue - a CNN


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 8
learning_rate = 1e-3
channels = 1
depth = 64


class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, dropout_p=0.3):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.norm1(self.features(x))
        out = self.act_layer(out)
        out = self.dropout(out)
        return out

class ConvNetwork(nn.Module):
    def __init__(self, dim, in_channels, num_classes=10, dropout_p=0.3):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling_depth = 2*self.dim

        self.block1 = BasicBlock(self.in_channels, self.dim, 3, stride=2, dropout_p=dropout_p)
        self.block2 = BasicBlock(self.dim, self.dim*2, 3, stride=2, dropout_p=dropout_p)

        self.ave_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(self.pooling_depth, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        out = self.ave_pool(x)
        out = out.view(out.size(0), -1) #flatten

        return self.classifier(out)


#Data
print("> Setup dataset")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.MNIST(root='/mnt/d/uqwdai1/data/rdm_backed/pytorch/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True) #num_workers=6

testset = torchvision.datasets.MNIST(root='/mnt/d/uqwdai1/data/rdm_backed/pytorch/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False) #num_workers=6


model = ConvNetwork(depth, channels, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
print("> Training")
start = time.time() #time generation
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


# Test the model
print("> Testing")
start = time.time() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


