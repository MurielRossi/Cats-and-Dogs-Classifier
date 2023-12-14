from collections import OrderedDict
import torch
import model as m
import inference

# Set the data directory, image directory, and model path
data_dir = '.\data\kagglecatsanddogs_5340\PetImages'
img_dir = 'images.jpg'
model_path = 'model.pth'

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and transform the data for training and testing
trainloader, testloader, train_transforms = m.load_ad_transform_data(data_dir, device)

# Train and test the model
model = m.train_and_test(trainloader, testloader, device)

# Save the trained model
torch.save(model, model_path)

print("End training ...")




