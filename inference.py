import torch
from PIL import Image
from torchvision import datasets, transforms, models

# Define the function for image prediction
def predict(model_path, img_dir):
    # Open the image using the Python Imaging Library (PIL)
    img = Image.open(img_dir)

    # Defines the same set of transformations applied during training
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(255),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    
    # Apply the transformations to the image
    img_normalized = transform(img).float()
    img_normalized = img_normalized.unsqueeze_(0)  # Add an extra dimension for batch size

    # Load the model
    model = torch.load(model_path)

    # Perform prediction without gradient computation
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        pred = model(img_normalized)  # Make a prediction
   
    return pred  # Return the prediction result
