# here I plan to laod in all my data and have my training sets etc

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

#becuase of the way Image folder works, since AD folder is first and NC second, AD gets label 0 and NC label 1
trainset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/train",
    transform=transform
)

testset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/test",
    transform=transform
)

train_loader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    testset,
    batch_size=100,
    shuffle=False
)

# also want holdoutset
#holdout_loader


print("Class mapping:", trainset.class_to_idx)
print("Training images:", len(trainset))
print("Testing images:", len(testset))




