# here I plan to laod in all my data and have my training sets etc
import torch
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import random
from collections import defaultdict

transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

#becuase of the way Image folder works, since AD folder is first and NC second, AD gets label 0 and NC label 1
full_trainset = ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/train",
    transform=transform
)

# Load the independent test dataset
testset = datasets.ImageFolder(
    root="/home/groups/comp3710/ADNI/AD_NC/test",
    transform=transform
)

# Desired mapping: AD = 1, NC = 0
# Apply this mapping to labels returned by the dataset.
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, 1 - label


full_trainset = RemappedDataset(full_trainset)
testset = RemappedDataset(testset)


# Group image indices by patient ID
patient_to_indices = defaultdict(list)

for index, (path, _) in enumerate(
    full_trainset.dataset.samples
):
    filename = os.path.basename(path)
    patient_id = filename.split("_", 1)[0]
    patient_to_indices[patient_id].append(index)

# Split patients into training and validation sets
patient_ids = list(patient_to_indices.keys())
random.Random(42).shuffle(patient_ids)

split_index = round(0.70 * len(patient_ids))

train_patients = set(patient_ids[:split_index])
validation_patients = set(patient_ids[split_index:])

# Convert patient groups into image indices
train_indices = [
    index
    for patient_id in train_patients
    for index in patient_to_indices[patient_id]
]

validation_indices = [
    index
    for patient_id in validation_patients
    for index in patient_to_indices[patient_id]
]

# Create datasets without copying any image files
trainset = Subset(full_trainset, train_indices)
validationset = Subset(full_trainset, validation_indices)

# Create dataloaders
train_loader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

validation_loader = DataLoader(
    validationset,
    batch_size=100,
    shuffle=False
)

test_loader = DataLoader(
    testset,
    batch_size=100,
    shuffle=False
)

# Verify patient separation
assert train_patients.isdisjoint(validation_patients)

print(f"Training images:   {len(trainset)}")
print(f"Validation images: {len(validationset)}")
print(f"Test images:       {len(testset)}")
print(f"Training patients: {len(train_patients)}")
print(f"Validation patients: {len(validation_patients)}")
print(f"Patient overlap: {len(train_patients & validation_patients)}")

