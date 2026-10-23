import time
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import train_loader, validation_loader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_vram = torch.cuda.get_device_properties(device).total_memory
print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"Total VRAM: {total_vram / (1024**3):.2f} GiB")

if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")


def test_accuracy(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader: #validation_loader
            images = images.to(device)
            labels = labels.to(device).long()

            # Get predictions for THIS validation batch
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


#Training

def train_model(model, data_loader, num_epochs):

    epochs = [x + 1 for x in range(num_epochs)]
    losses = []
    test_accuracies = []

    print("> Training")
    start = time.time()

    # Reset peak GPU memory statistics
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(num_epochs):
        model.train()

        loss = None
        for i, (images, labels) in enumerate(data_loader): #train_loader
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(data_loader)}], "
                    f"Loss: {loss.item():.5f}"
                )
        if (loss is not None):
            losses.append(loss.item())
        else:
            print("something dodgy")
        test_accuracies.append(test_accuracy(model, validation_loader))

    elapsed = time.time() - start

    # Peak GPU memory used during training
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory allocated: {peak_allocated / (1024**2):.2f} MiB")

    print(f"Training took {elapsed:.2f} seconds "
        f"({elapsed / 60:.2f} minutes)")


    return(epochs, losses, test_accuracies)

# Validation / Evaluation
# --------------------

def validate_model(model, data_loader):

    print("> Evaluating on validation set")
    start = time.time()

    model.eval()

    total_AD = 0
    total_NC = 0
    correct_AD = 0
    correct_NC = 0
    correct = 0
    total = 0
    false_negative = 0
    false_positive = 0

    actual_labels = []
    predicted_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            # Get predictions for THIS validation batch
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            # Collect data for AUROC
            actual_labels.extend(labels.cpu().tolist())
            predicted_probs.extend(
                probabilities[:, 1].cpu().tolist()
            )

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class masks
            AD_mask = labels == 1
            NC_mask = labels == 0

            total_AD += AD_mask.sum().item()
            total_NC += NC_mask.sum().item()

            # Correct predictions
            correct_AD += (
                (predicted == 1) & AD_mask
            ).sum().item()

            correct_NC += (
                (predicted == 0) & NC_mask
            ).sum().item()

            # Incorrect predictions
            false_negative += (
                (predicted == 0) & AD_mask
            ).sum().item()

            false_positive += (
                (predicted == 1) & NC_mask
            ).sum().item()

    elapsed = time.time() - start
    print(f"Evaluation took {elapsed:.2f} seconds "
      f"({elapsed / 60:.2f} minutes)")

    # Calculate metrics safely
    accuracy = correct / total if total > 0 else 0

    precision_denominator = correct_AD + false_positive
    recall_denominator = correct_AD + false_negative

    precision = (
        correct_AD / precision_denominator
        if precision_denominator > 0 else 0
    )

    recall = (
        correct_AD / recall_denominator
        if recall_denominator > 0 else 0
    )

    f1_denominator = precision + recall
    f1_score = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0 else 0
    )

    print(f"Validation Accuracy: {100 * accuracy:.2f}%")
    print(f"Precision (AD):      {100 * precision:.2f}%")
    print(f"Recall (AD):         {100 * recall:.2f}%")
    print(f"F1 Score (AD):       {f1_score:.4f}")
    print(f"True Positives:      {correct_AD}")
    print(f"True Negatives:      {correct_NC}")
    print(f"False Positives:     {false_positive}")
    print(f"False Negatives:     {false_negative}")
    print(f"Total AD images:     {total_AD}")
    print(f"Total NC images:     {total_NC}")

    return (actual_labels, predicted_probs)

def calculateLatency(model, data_loader):
    model.eval()
    images, _ = next(iter(data_loader))
    single_image = images[:1].to(device)

    warmup_iterations = 20
    timed_iterations = 100

    with torch.no_grad():
        # Warm up the model before timing
        for _ in range(warmup_iterations):
            _ = model(single_image)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(timed_iterations):
                _ = model(single_image)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / timed_iterations) * 1000


#script to run it all

# Hyperparameters
num_epochs = 50
learning_rate = 1e-3
channels = 1
depth = 64

model = None # this is where you would configure your imported model 

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train the model

(epochs, losses, test_accuracies) = train_model(model, train_loader, num_epochs)


#test the model
(actual_labels, predicted_probs) = validate_model(model, validation_loader)

#pickle dump for later
payload = (epochs, losses, test_accuracies, actual_labels, predicted_probs)
with open('data.pickle', 'wb') as file:
    pickle.dump(payload, file)

# latency statistic
print(f"Average inference latency: {calculateLatency(model, validation_loader):.3f} ms")
