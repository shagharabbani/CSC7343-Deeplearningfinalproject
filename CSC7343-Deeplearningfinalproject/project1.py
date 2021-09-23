
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


class Model(nn.Module):
    def __init__(self):
        """
        docstring
        """

        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(485, 60, kernel_size=2, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(60, 1, kernel_size=2, stride=1),
            nn.ReLU())

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
         out = self.layer1(x)
         out = self.layer2(out)
         return out 



    def train(self,x_train,y_train):
        """
        Train the model based on x as input and y as output
        """

        x_train = torch.from_numpy(x_train).float()
        x_train = torch.unsqueeze(x_train,0)
        # x_train = torch.unsqueeze(x_train,0)

        y_train = torch.from_numpy(y_train).float()
        y_train = torch.unsqueeze(y_train,0)
        # y_train = torch.unsqueeze(y_train,0)

        #Fitting the training data to the network
        acc_list = [] 
        ciccio=[]
        for x in range(len(y_train)):

            for i in range(10):
                # Run the forward pass
                imp, seis = x_train,y_train
                outputs = self(imp)
                loss = self.criterion(outputs, seis)

                # Backprop and perform Adam optimisation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ciccio.append(loss.data.numpy())  
                
                if (i + 1) % 100 == 0:  
                    print(x)




    def predict(self,x):
        """
        Predict the label of x
        """
        y_pred = self(x)


        return y_pred

