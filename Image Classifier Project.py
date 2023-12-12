#!/usr/bin/env python
# coding: utf-8

# ## Prepare the workspace

# In[23]:


# Before you proceed, update the PATH
# Arc
import os

os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"
# Restart the Kernel at this point. 


# In[24]:


# Do not execute the commands below unless you have restart the Kernel after updating the PATH. 
get_ipython().system('python -m pip install torch')


# In[25]:


# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 


# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[26]:


# Imports here
import os
import json 
import time 
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

import torch
from torch import nn
from torch import optim

from tqdm import tqdm


from torch.autograd import Variable
from torchvision import datasets, transforms,models
from collections import OrderedDict
import torch.nn.functional as F


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). 

# If you do not find the `flowers/` dataset in the current directory, **/workspace/home/aipnd-project/**, you can download it using the following commands. 
# 
# ```bash
# !wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
# !unlink flowers
# !mkdir flowers && tar -xzf flower_data.tar.gz -C flowers
# ```
# 

# ## Data Description
# The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[27]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[28]:


# lets define the means and std deviations for normalization of our data
means_data = [0.485, 0.456, 0.406]
stds_data = [0.229, 0.224, 0.225]
# so the train transfrom part :
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means_data, stds_data)
])

#  SO the validation parts and testing, only resize 
valid_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(means_data, stds_data)
])

# So for the test part 
test_transform = valid_test_transform

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': train_transform,
    'valid': valid_test_transform,
    'test': test_transform
}

# Define the data loaders
# Load the datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transform)
# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': train_dataset,
    'valid': valid_dataset,
    'test': test_dataset
}
# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
}


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[29]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

cat = len(cat_to_name)


# In[30]:


print(cat_to_name)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# ## Note for Workspace users: 
# If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[32]:


# lets used model vgg for the start
model=models.vgg16(pretrained=True)

#DONE: Param part
for param in model.parameters():
    param.requires_grad=False

print(model.classifier)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[33]:


# defines the classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(hidden_units, cat)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = torch.nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# replaces the pretrained classifier with the one created above
model.classifier = classifier

model.classifier


# In[35]:


# Replace the classifier in the pre-trained model
model.classifier = classifier

# Set hyperparameters
num_epochs = 3
learning_rate = 0.001  # You can adjust this value
hidden_units = 512
print_every = 16

# Move model to GPU if available
model.to(device)

# Define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs, labels in tqdm(dataloaders['train'], desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print training metrics after each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}.. '
          f'Training Loss: {running_loss/len(dataloaders["train"]):.3f}')


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[36]:


# Initialize test accuracy
total_test_accuracy = 0

# Start the timer for testing
start_time_testing = time.time()
print('Testing started.')

# Perform testing on the test set
with torch.no_grad():  
    for batch_idx, (test_images, test_labels) in tqdm(enumerate(test_loader, 1), desc='Testing Progress', unit='batch'):
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_logits = model.forward(test_images)
        test_probs = torch.exp(test_logits)
        top_test_probs, top_test_class = test_probs.topk(1, dim=1)
        matches = (top_test_class == test_labels.view(*top_test_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()
        total_test_accuracy += accuracy.item()  # Accumulate accuracy

# Stop the timer for testing
end_time_testing = time.time()
print('Testing completed.')
testing_time = end_time_testing - start_time_testing
print(f'Testing time: {testing_time:.0f}s')

# Calculate and print the average test accuracy
average_test_accuracy = total_test_accuracy / len(test_loader)
print(f'Average Test Accuracy: {average_test_accuracy * 100:.2f}%')


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[57]:


# Save Checkpoint
epochs = 3
checkpoint = {
    'classifier': model.classifier,  
    'state_dict': model.state_dict(),
    'learning_rate': learning_rate,
    'class_to_idx': train_dataset.class_to_idx,
    'optimizer_dict': optimizer.state_dict()
}

torch.save(checkpoint, 'checkpoint_test.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[58]:


def load_checkpoint(path='checkpoint.pth'):
    # Load the saved file
    checkpoint = torch.load(path, map_location=device)

    # Download pretrained model
    model = models.vgg16(pretrained=True)

    # To freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load from checkpoint
    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

loaded_model = load_checkpoint('checkpoint_test.pth' )


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[59]:


import os

# Specify the path to the directory
image_directory = data_dir + '/test' + '/15/'
image_directory = data_dir + '/train' + '/1/'
test = "flowers/train/1/"
# List all files in the directory
image_files = os.listdir(image_directory)

# Print the names of the images
for image_file in image_files:
    print(image_file)


# In[68]:


# Image Processing Parameters Project
project_px_size = 256
project_px_crop = 224

def process_image(test_image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a NumPy array
    '''
    # Open image and read width and height of it
    test_image = Image.open(test_image_path)
    width, height = test_image.size

    # Calculate new width and new height depending upon which side is shorter
    if width > height:
        new_height = project_px_size
        if new_height > height:
            new_width = int(width * (height / new_height))
        else:
            new_width = int(width * (new_height / height))
    else:
        new_width = project_px_size
        if new_width > width:
            new_height = int(height * (width / new_width))
        else:
            new_height = int(height * (new_width / width))

    # Resize image
    test_image = test_image.resize((new_width, new_height))

    # Crop square of 224x224 px from center
    test_image = test_image.crop(((new_width - project_px_crop) // 2, (new_height - project_px_crop) // 2,
                                  (new_width + project_px_crop) // 2, (new_height + project_px_crop) // 2))

    # Create NumPy-Array from Image
    np_image = np.array(test_image)

    # Do Normalization
    channels_means = [0.485, 0.456, 0.406]
    channels_stds = [0.229, 0.224, 0.225]

    np_image = np_image / 256
    np_image = (np_image - channels_means) / channels_stds

    # Transpose ColorChannel
    np_image = np_image.transpose((2, 0, 1))

    return np_image

# Example usage
image_path = 'flowers/train/1/image_06734.jpg'
processed_image = process_image(image_path)
print(processed_image.shape)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[61]:


def imshow(image_tensor, axis=None, title=None):
    """
    Display a PyTorch tensor as an image using Matplotlib.

    Parameters:
        image_tensor (torch.Tensor): Input image tensor.
        axis (matplotlib.axes._axes.Axes, optional): Matplotlib axis to use. If None, a new figure is created.
        title (str, optional): Title for the displayed image.

    Returns:
        matplotlib.axes._axes.Axes: Matplotlib axis used for plotting.
    """
    if axis is None:
        fig, axis = plt.subplots()

    # Convert PyTorch tensor to NumPy array and move channels to the last dimension
    image_np = image_tensor.cpu().numpy().transpose((1, 2, 0))

    # Undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    image_np = std_dev * image_np + mean

    # Clip the values to be in the range [0, 1]
    image_np = np.clip(image_np, 0, 1)

    # Display the image
    axis.imshow(image_np)

    if title:
        axis.set_title(title)

    return axis

# Example usage
imshow(processed_image, title='Processed Image')
plt.show()


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[63]:


def predict(image_path, model, topk=5, device='cuda'):
    model.eval()
    model.to(device)

    image = process_image(image_path)
    image = torch.FloatTensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)

    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk, dim=1)

    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices.cpu().numpy().flatten()]

    return top_probabilities.cpu().numpy().flatten(), top_classes

# Example usage
model_path = 'checkpoint_test.pth'
image_path = 'flowers/train/1/image_06736.jpg' 

model = load_checkpoint(model_path)
probs, classes = predict(image_path, model)

print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[ ]:


def display_prediction(image_path, model, cat_to_name):
    # Set up plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    # Find Flower Name
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name.get(flower_num, flower_num)
    plt.title(title_)

    # Preprocess Image
    test_image = process_image(image_path)
    img = Image.open(image_path)
    plt.imshow(img)

    # Make Prediction
    model.to(device)
    model.eval()

    with torch.no_grad():
        image_tensor = torch.from_numpy(test_image).float().unsqueeze(0).to(device)
        log_probs = model(image_tensor)
        probs = torch.exp(log_probs)

    # Get top 5 predictions
    top_probs, top_classes = probs.topk(5, dim=1)
    top_probs, top_classes = top_probs.cpu().numpy()[0], top_classes.cpu().numpy()[0]

    # Convert class indices to flower names
    top_flower_names = [cat_to_name.get(str(cls), str(cls)) for cls in top_classes]

    # Display top 5 predictions
    data_probs = {'Probabilities': top_probs * 100}
    df = pd.DataFrame(data_probs, index=top_flower_names)

    # Plot bar chart
    plt.subplot(2, 1, 2)
    plt.xlabel('Probability in %')
    plt.ylabel("Flowers")
    base_color = sb.color_palette()[0]
    sb.barplot(x=df['Probabilities'], y=df.index, color=base_color)

    plt.show()

# Example usage
model_path = 'checkpoint_test.pth'
image_path = 'flowers/train/1/image_06736.jpg'
display_prediction(image_path, load_checkpoint(model_path), cat_to_name)


# ## Reminder for Workspace users
# If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
#     
# We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.

# In[ ]:


# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace


# In[28]:


# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Training parameters
epochs = 10
print_every = 20

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print statistics
        if i % print_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i}/{len(train_loader)}], Loss: {running_loss / print_every:.4f}")
            running_loss = 0.0

# Save the trained model
torch.save(net.state_dict(), 'flower_classifier.pth')
print("Training complete. Model saved.")


# In[ ]:




