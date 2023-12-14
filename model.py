import os
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# Define a function to load and transform the data
def load_ad_transform_data(data_dir, device):
    
    # Define image transformations for training and testing
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load training and testing data with specified transformations
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # Create DataLoader objects for training and testing
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128)

    return trainloader, testloader, train_transforms

# Define a function for training and testing the model
def train_and_test(trainloader, testloader, device):
    # Load a pre-trained DenseNet model
    model = models.densenet121(pretrained=True)
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Customize the classifier for binary classification
    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 2),
                                     nn.LogSoftmax(dim=1))

    # Move the model to the specified device (e.g., GPU)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.00005)

    # Initialize variables for training loop
    accuracy = 0
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5
    max_accuracy = 0
    
    # Training loop
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            if accuracy == 1:
                break
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            
            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print and evaluate the model every few steps
            if steps % print_every == 0:
                if accuracy == 1:
                    break
                test_loss = 0
                accuracy = 0
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print the model's performance metrics
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"LossF: {test_loss:.3f}.. "
                      f"accuracyF: {accuracy:.3f}.. "
                      f"Test loader: {len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                
                # Save the model whenever performance improves
                if max_accuracy < accuracy:
                    max_accuracy = accuracy
                    torch.save(model, 'model2.pth')

                running_loss = 0
                model.train()
    
    # Return the trained model
    return model
