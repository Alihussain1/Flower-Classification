import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import json
import torchvision.models as models
import time
from PIL import Image
import math
import cv2

def create_dataloaders(data_dir,batch_size = 64 , mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225), sampler = None):
    data_transform = transforms.Compose([    
        transforms.RandomRotation(50),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    dataset = datasets.ImageFolder(data_dir,data_transform)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,sampler=sampler)
    return loader
def create_model(model_name = "alexnet"):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        first_layer_in = 9216
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        first_layer_in = 2048
    else:
        return 
    
    for param in model.parameters():
        param.requires_grad = False
    
    if model_name == "alexnet":
        model.classifier = nn.Sequential(#nn.Dropout(0.5),
                                nn.Linear(first_layer_in,4096),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(4096,2048),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(2048,102),
                                 nn.LogSoftmax(dim=1)
                                )
        # specify loss function
        criterion = nn.NLLLoss()
        # specify optimizer
        optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
    elif model_name == "resnet50":
        model.fc = nn.Sequential(
                                nn.Linear(first_layer_in,4096),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(4096,2048),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(2048,102),
                                 nn.LogSoftmax(dim=1)
                                )
        # specify loss function
        criterion = nn.NLLLoss()
        # specify optimizer
        optimizer = optim.Adam(model.fc.parameters(),lr=0.001)
    return model , criterion, optimizer
def train_model(model, device, train_loader, valid_loader, criterion, optimizer, epochs = 200):
    valid_loss_min = np.inf
    model.to(device)
    for i in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        total_correct = 0
        model.train()
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        #if i % 10 == 0 :
        elabsed_time = time.time() - start_time
        #elabsed_time /= (60 * 10**7) 
        model.eval()
        for images, labels in valid_loader:            
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():               
                predictions = model(images)        
                loss = criterion(predictions,labels)
                valid_loss += loss.item() * images.size(0)
                _, pred = torch.max(predictions, 1)
                # compare predictions to true label
                total_correct += pred.eq(labels.data.view_as(pred)).sum(dim=0)

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        print('Epoch: {} \tElabsed Time : {} Seconds\nTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tValidation Accuracy: {}%'.format(
                i+1, elabsed_time,train_loss, valid_loss, total_correct*100/len(valid_loader.dataset) ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), 'model.pt')
def process_image(image):
    if image.size[0] > image.size[1]:
        new_size = ( math.ceil((image.size[0]*256) / image.size[1]),256)
    else:
        new_size = ( 256,math.ceil((image.size[1]*256) / image.size[0]))
    image = image.resize(new_size)
    center_point = (math.ceil(image.size[0]/2),math.ceil(image.size[1]/2))
    area = (center_point[0]-112,center_point[1]-112,center_point[0]+112,center_point[1]+112)
    image = image.crop(area)
    np_image = np.array(image)
    np_image_normalized = cv2.normalize(np_image.astype(float),None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_normalized -= mean
    np_image_normalized /= std
    np_image_normalized = np_image_normalized.transpose(2,0,1)
    return np_image_normalized
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
def predict(image_path, model,device,cat_to_name,topk=5):
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.float()
    image = image.reshape((1,3,224,224))
    model.eval()
    with torch.no_grad():
        prediction = model(image.to(device))
    model.train()
    prediction_softmax = F.softmax(prediction.data,dim=1)
    top_predictions = prediction.topk(topk)
    top_predictions_softmax = prediction_softmax.topk(topk)
    classes = top_predictions_softmax[1][0].cpu().numpy()
    classes_names = [cat_to_name[str(c+1)] for c in classes] 
    prop = top_predictions_softmax[0][0].cpu().numpy()
    classes *= 100
    
    fig, ax = plt.subplots(1,2,figsize=(20,8))
    imshow(image[0],ax=ax[0])
    y_pos = np.arange(topk)
    plt.barh(y_pos, prop, align='center', alpha=0.5)
    plt.yticks(y_pos, classes_names)
    plt.xlabel('Probabilities')
    plt.title('top {} probabilities'.format(topk))
    plt.show()