from torch import nn
import torch
from matplotlib import plt




def da_a(data):  # this function turns a numpy vector into a torch tensor of the right dimensions
     ciccio1 = torch.from_numpy(data).float()
     ciccio2 = torch.unsqueeze(ciccio1,0)
     ciccio3 = torch.unsqueeze(ciccio2,0)
     return ciccio3


num_epochs = 100
learning_rate = 0.001

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 60, kernel_size=2, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(60, 1, kernel_size=2, stride=1),
            nn.ReLU())

    def forward(self, x):
         out = self.layer1(x)
         out = self.layer2(out)
         return out  

model = ConvNet() 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


acc_list = [] 
ciccio=[]
for x in range(num):

    for i in range(num_epochs):
        # Run the forward pass
        imp, seis = da_a(imp_p[:,x]), da_a(data_sim[:,x])
        outputs = model(imp)
        loss = criterion(outputs, seis)

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ciccio.append(loss.data.numpy())  
        
        if (i + 1) % 100 == 0:  
            print(x)

 # now i apply the net to a model that wasn't used before: imp_true

Imp_true=da_a(imp_true)
outputs = model(Imp_true)
mat1=outputs.detach().numpy()
plt.subplot(1,2,1)
plt.plot(mat1.flatten())
plt.subplot(1,2,2)
plt.plot(ciccio)
plt.show()