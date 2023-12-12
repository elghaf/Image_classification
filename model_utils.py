# model_utils.py
import torch
from torchvision import models
from torch import nn, optim
from data_utils import load_data
from tqdm import tqdm


from collections import OrderedDict
def build_model(arch='vgg16', learning_rate=0.01, hidden_units=512, gpu=False):
    
    # Check if the specified architecture is supported
    supported_architectures = ['vgg16', 'densenet121']
    if arch not in supported_architectures:
        raise ValueError("Unsupported architecture. Please choose from 'vgg16' or 'densenet121'.")

    # Load a pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Freeze feature parameters
        for param in model.parameters():
            param.requires_grad = False
        # Classifier for VGG16
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze feature parameters
        for param in model.parameters():
            param.requires_grad = False
        # Classifier for Densenet
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        raise ValueError("Unsupported architecture. Please choose from 'vgg16' or 'densenet121'.")

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
        'arch': arch,
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)
    print("model check is saved succ")

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def predict(model, image, topk=5, gpu=False):
    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process image and convert to a 1D tensor
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)

    top_probs, top_indices = torch.topk(probabilities, topk)
    top_probs = top_probs.cpu().numpy().tolist()[0]
    top_indices = top_indices.cpu().numpy().tolist()[0]

    return top_probs, top_indices

