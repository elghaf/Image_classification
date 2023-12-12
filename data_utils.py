# data_utils.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_data(data_dir):
    # Data transforms
    means_data = [0.485, 0.456, 0.406]
    stds_data = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means_data, stds_data)
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means_data, stds_data)
    ])

    test_transform = valid_test_transform

    # Load datasets
    train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
    valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=valid_test_transform)
    test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    return dataloaders, train_dataset.class_to_idx

def process_image(image_path):
    # Load and process the image
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image)

    return image
