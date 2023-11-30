# model_utils.py
import torch
from torchvision import models
from torch import nn, optim
from data_utils import load_data
from tqdm import tqdm

def build_model(arch='vgg16', learning_rate=0.01, hidden_units=512, gpu=False):
    # Load a pre-trained model
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Define the criterion (loss function)
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move the model to GPU if available
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, data_dir, epochs=20, gpu=False):
    # Load the data
    dataloaders, _ = load_data(data_dir)

    # Move the model to GPU if available
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    for epoch in range(epochs):
        # Training the model
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f'{total_loss/(progress_bar.n + 1e-5):.3f}')  # Update the progress bar with the average loss

        # Print the training statistics
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training loss: {total_loss/(len(dataloaders['train'])):.3f}")


def save_checkpoint(model, optimizer, save_dir):
    checkpoint_path = f'{save_dir}/checkpoint.pth'

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
