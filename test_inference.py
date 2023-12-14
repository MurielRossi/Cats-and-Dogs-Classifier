import inference
from torchvision import datasets, transforms, models
import os
import re

# Set paths and directories
model_path = 'model2.pth'
data_dir = '.\data\kagglecatsanddogs_5340\PetImages'
inference_img_dir = '.\images'
array_labels = ["Cat", "Cat", "Cat", "Dog", "Dog", "Dog", "Dog", "Dog", "Dog", "Dog", "Cat", "Cat", "Cat", "Cat"]

# Create an empty list to store image paths
array_img = []

# Iterate through files in the inference image directory and append their paths to the list
for file in os.listdir(inference_img_dir):
    path_img = os.path.join(inference_img_dir, file)
    array_img.append(path_img)
    
# Define image transformations for inference
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Iterate through the list of image paths
for i, img_dir in enumerate(array_img):
    # Perform prediction using the inference module
    pred = inference.predict(model_path, img_dir)

    # Extract the predicted class with the highest probability
    index = pred.data.cpu().numpy().argmax()
    train_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    classes = train_data.classes
    class_name = classes[index]

    # Extract the ground truth label from the image file name
    pattern = re.compile(r'(Cat|Dog)')
    match = pattern.search(img_dir)

    # Print the prediction results
    print("\nCrude prediction {}: ".format(i+1), pred)
    print("Prediction: ", class_name, " Label: ", match.group())
