import torch
import numpy
import pandas
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_folder_path : str = "test"
test_folder_path : str= "train"


torch.set_printoptions(precision = 10)

def normalise(t: torch.tensor):
    max: float = t.max()
    min: float = t.min()
    t =  ((t - min) / (max - min)) #implicit broadcasting applied on scalars
    return t

def parse_command_line():
    i = 1
    print(sys.argv)
    while i < len(sys.argv):
        if sys.argv[i] == "-t":
            i += 1
            train_folder_path = sys.argv[i]
        elif sys.argv[i] == "-T":
            i += 1
            test_folder_path = sys.argv[i]
        i += 1    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features= 30276, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=5)
        #note hyperparameter choice is arbitrary except initial in and final out
        #they are dependant on the colour channels (3 since 3 GCMs) and output classes (5 since 5 classes on cat5) respectively
        
        
    def forward(self, t):
    # implement the forward pass
    
        # (1) input layer
        t = t #usually omitted since this is obviously trivial; size 360*131

        # (2) hidden conv layer
        t = self.conv1(t) #Haven't implemented wrapping - so after a 5x5 convolution, discard borders meaning feature maps are now 6 * 127 * 356 (Channels * height * width)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=2, stride=2)
        #pooling 2x2 with stride 2 - reduces to 6 * 178 * 63

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=2, stride=2)
        #pooling 2x2 with stride 2 - reduces to 12 * 29 * 87

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 29 * 87)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1) #implicitly performed by F.cross_entropy()

        return t


parse_command_line()
print(train_folder_path)

targetFilePath : str = "data/Target_TMean_NNI_regional_ave_time_series.csv"

#reading in labels from target CSV
labels = pandas.read_csv(targetFilePath, sep=",", skiprows=1, header=None)
labels = labels[labels.columns[3]].astype(int) #cat5 column - so 5 prediction classes

#Reading in GCM CSVs with pandas
precip = pandas.read_csv("data/PRECIP_1993_2016_ecmwf.csv", sep=",", skiprows=3, header=None)
precip = precip.drop(precip.columns[0], axis=1)

t2m = pandas.read_csv("data/T2M_1993_2016_ecmwf.csv", sep=",", skiprows=3, header=None)
t2m = t2m.drop(t2m.columns[0], axis=1)

z850 = pandas.read_csv("data/Z850_1993_2016_ecmwf.csv", sep=",", skiprows=4, header=None)
z850 = z850.drop(z850.columns[0], axis=1)

#converting to pytorch tensors
labelsTensor = (torch.from_numpy(labels.values).type(torch.LongTensor) + 2).cuda() #cat5 in form -2, -1, 0, 1, 2; we want 0 to 5 for brevity
precip = torch.from_numpy(precip.values).type(torch.FloatTensor)
t2m = torch.from_numpy(t2m.values).type(torch.FloatTensor)
z850 = torch.from_numpy(z850.values).type(torch.FloatTensor)

#Unflattening GCMs
precip = precip.reshape(288, 131, 360)
t2m = t2m.reshape(288, 131, 360)
z850 = z850.reshape(288, 131, 360)

#normalising GCMs
precip = normalise(precip)
t2m = normalise(t2m)
z850 = normalise(z850)

#Stacking each GCM of each variable as a channel of the input tensor
dataTensor = torch.stack([precip, t2m, z850], dim=1).cuda()

#Code for plotting input as greyscale images and combining into a single RGB image
"""pyplot.imshow(precip[0], cmap="gray")
pyplot.show()
pyplot.imshow(t2m[0], cmap="gray")
pyplot.show()
pyplot.imshow(z850[0], cmap="gray")
pyplot.show()
pyplot.imshow(dataTensor[0].cpu().permute(1, 2, 0))
pyplot.show()"""

#Split the dataTensor into training and test tensors for explicit holdout
trainTensor = dataTensor[:250]
testTensor = dataTensor[250:] #38/288 for testing, or about 13.2%

#Creating pytorch dataset and dataloader for easy access to minibatch sampling without replacement in randomnised order
trainset = torch.utils.data.TensorDataset(trainTensor, labelsTensor[:250])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
testset = torch.utils.data.TensorDataset(testTensor, labelsTensor[250:])
testloader = torch.utils.data.DataLoader(testset, batch_size=38, shuffle=False)

#Initialising the CNN and gradient descender (optimizer)
network = Network()
network.cuda()
optimizer = optim.SGD(network.parameters(), lr = 0.01)

epochLossSum = 0
for epoch in range(0, 250):
    batchNo = 0
    epochCorrect = 0
    prevloss = epochLossSum
    epochLossSum = 0

    for images, labels in trainloader: #shuffled minibatches
        batchNo += 1
        preds0 = network(images)
        loss0 = F.cross_entropy(preds0, labels)  #loss on this batch before gradient descent

        optimizer.zero_grad()
        loss0.backward() #this call updates the gradients in network
        optimizer.step()
        preds1 = network(images)
        loss1 = F.cross_entropy(preds1, labels) #loss on this batch after gradient descent (expect decrease)
        
        #nb loss0, preds0 are loss and predictions before training on this batch, loss1, pred1 are after training on this batch (needed to see how much of a difference this batch made)

        epochLossSum += loss0.item()
        epochCorrect += preds0.argmax(dim=1).eq(labels).int().sum().item()
        #print("Minibatch ", batchNo, ":\t", round(loss0.item(), 5), "(before gd)\t", round(loss1.item(), 5), "(after gd)\t", preds1.argmax(dim=1).eq(labels).int().sum().item(), " (correct predictions)", sep = "" )

    print("epoch", epoch, "\tloss: ", round(epochLossSum,5), "\tdifference: " , round(epochLossSum - prevloss, 5) , "\t%correct:", round((100 * epochCorrect / 250), 3), sep="")
    if (epochCorrect / 250 * 100) > 99:
        break

for images, labels in testloader:
    preds = network(images)
    correct = preds.argmax(dim=1).eq(labels).int().sum().item()
    #print(testpreds.argmax(dim=1))
    #print(labels)
    print("result:", correct, "/ 38")
    print("test correct %:", 100 * correct / 38.0)

    '''
    confusionStack = torch.stack((labels, preds.argmax(dim=1)), dim=1)
    confusionMatrix = torch.zeros(5, 5, dtype=torch.int64).cuda()
    for p in confusionStack:
        tl, pl = p.tolist()
        confusionMatrix[tl, pl] = confusionMatrix[tl, pl] + 1
    pyplot.figure(figsize=(10,10))
    plot_confusion_matrix(confusionMatrix.cpu(), ('-2', '-1', '0', '1', '2'))
    pyplot.show()
    '''
print("END")

