
import pandas
# import torch
import numpy
# import cv2


# importing the libraries
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# transformations to be applied on images
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                            ])


class Model:
    def __init__(self):
        """
        docstring
        """

        from sklearn.neural_network import MLPClassifier, Classifier
        self.classifier = MLPClassifier(hidden_layer_sizes=(150,1000,50), 
                           max_iter=500,activation = 'relu',solver='adam',random_state=1)
        # pass

    def train(self,x,y):
        """
        Train the model based on x as input and y as output
        """

        #Fitting the training data to the network
        self.classifier.fit(x, y)




    def predict(self,x):
        """
        Predict the label of x
        """
        y_pred = self.classifier.predict(x)


        return y_pred


class ModelPyTorch(nn.Module):
    def __init__(self):
        """
        docstring
        """
        
        super(ModelPyTorch, self).__init__()

        self.layer1 = torch.nn.Conv1d(in_channels=7, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)


        # self.cnn_layers = nn.Sequential(
        #     # Defining a 1D convolution layer
        #     nn.Conv1d(1441, 1, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     # Defining another 1D convolution layer
        #     nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        # )

        # self.linear_layers = nn.Sequential(
        #     nn.Linear(4 * 7 * 7, 10)
        # )

        

    # Defining the forward pass    
    def forward(self, x):
        # x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        # return x

        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)

        log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return log_probs




    def train(self,x,y):
        """
        Train the model based on x as input and y as output
        """

        x = torch.from_numpy(x).float()
        
        y = torch.from_numpy(y).float()
        
        # x,y=x.double(),y.double()
        x=x.unsqueeze(0).unsqueeze(0)
        y=y.unsqueeze(0).unsqueeze(0)

        # x = x.view(x.size(0), -1)
        # y = y.view(x.size(0), -1)

        self.optimizer=optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        for i in range(10):
            running_loss = 0

            for j in range(len(x)):
                images=x[j]
                labels=y[j]

                # if torch.cuda.is_available():
                #     images = images.cuda()
                #     labels = labels.cuda()

                # Training pass
                self.optimizer.zero_grad()
                
                output = self(images)
                loss = self.criterion(output, labels)
                
                #This is where the model learns by backpropagating
                loss.backward()
                
                #And optimizes its weights here
                self.optimizer.step()
                
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(y)))




    def predict(self,x):
        """
        Predict the label of x
        """
        y_pred = self(x)


        return y_pred
        
