import torch
import numpy
import pandas
import sys
import os
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Global option defaults that can be changed later by command line
gcm_folder_path : str = "gcms"
target_folder_path : str = "targets"
class_index = "cat5"

use_cuda : bool = False

train_split: float = 0.8
test_split: float = 0.1
validation_split: float = 0.1

batch_size: int = 10

max_training_epochs: int = 200

CMD_HELP : str = """--placeholder--"""

torch.set_printoptions(precision = 10)

def normalise(t: torch.tensor):
    max: float = t.max()
    min: float = t.min()
    t =  ((t - min) / (max - min)) #implicit broadcasting applied on scalars
    return t

def parse_command_line():
    i = 1 #sys.argv[0] contains the script name itself and can be ignored
    while i < len(sys.argv):
        if sys.argv[i] == "-h" or sys.argv[i] == "--help":
            print(CMD_HELP)
            sys.exit()
        elif sys.argv[i] == "--gcms-path":
            i += 1
            global gcm_folder_path
            gcm_folder_path = sys.argv[i]
        elif sys.argv[i] == "--classlabel":
            i += 1
            global class_index
            class_index = sys.argv[i]
        elif sys.argv[i] == "--cuda":
            global use_cuda
            use_cuda = True
        elif sys.argv[i] == "--targets-path":
            i += 1
            global target_folder_path
            target_folder_path = sys.argv[i]
        elif sys.argv[i] == "--test-percentage":
            i += 1
            global test_split
            test_percentage = float(sys.argv[i]) / 100.0
        elif sys.argv[i] == "--validation-percentage":
            i += 1
            global validation_split
            validation_percentage = float(sys.argv[i]) / 100.0
        elif sys.argv[i] == "--batch-size":
            i += 1
            global batch_size
            batch_size = int(sys.argv[i])
        elif sys.argv[i] == "--max-epochs":
            i += 1
            global max_training_epochs
            max_training_epochs = int(sys.argv[i])
        else:
            print("Unknown argument: " + sys.argv[i] + "\n Use \"gcm-cnn -h\" to see valid commands")
            sys.exit()
        i += 1
    global train_split
    train_split = 1.0 - test_split - validation_split
    assert(train_split > 0), "No instances left for training. Did the sum of your test and validation holdout percentages exceed 100%?"
    assert(batch_size > 0), "Batch size can't be negative!!!"

def read_gcm_folder(path: str): #returns a folder of GCM CSVs as a 4-channel PyTorch Tensors
    filenames = os.listdir(path)
    files = []
    for i in range(0, len(filenames)):
        nextfile =  pandas.read_csv((path + "/" + filenames[i]), sep=",", skiprows=3, header=None) #explicitly skip 3 rows to discard header, longitude, latitude
        nextfile = nextfile.drop(nextfile.columns[0], axis=1)
        nextfile = torch.from_numpy(nextfile.values).type(torch.FloatTensor)
        if use_cuda == True:
            nextfile = nextfile.cuda()
        nextfile = nextfile.reshape(288,131,360)
        nextfile = normalise(nextfile)
        files.append(nextfile)
        
    return torch.stack(files, dim=1)

def read_target_folder(path: str): #returns a folder of CSVs containing the class label as a list of PyTorch Tensors
    filenames = os.listdir(path)
    files = []
    for i in range(0, len(filenames)):
        nextfile = pandas.read_csv((path + "/" + filenames[i]), sep=",")
        nextfile = nextfile[class_index] + 2
        nextfile = torch.from_numpy(nextfile.values).type(torch.LongTensor)
        if use_cuda == True:
            nextfile = nextfile.cuda()
        files.append(nextfile)

    return files

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=len(os.listdir(gcm_folder_path)), out_channels=6, kernel_size=5)
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

#Setting options from command line
parse_command_line()

#print(target_tensors[0].size()[0])
#Reading files from disk into PyTorch tensors
label_tensors = read_target_folder(target_folder_path)
gcm_tensor = read_gcm_folder(gcm_folder_path)

#Split the gcm_tensor into train, validation, test tensors
instances = gcm_tensor.size()[0]
train_tensor = gcm_tensor[:int(instances * train_split)] #note int() truncates/floors
validation_tensor = gcm_tensor[int(instances * train_split):int(instances * (train_split + validation_split))]
test_tensor = gcm_tensor[int(instances * (train_split + validation_split)):]



#Now we set up a loop to train a network for each label file that was present
for n in range(0, len(label_tensors)):
    
    #Creating pytorch dataset and dataloader for easy access to minibatch sampling without replacement in randomnised order
    train_set = torch.utils.data.TensorDataset(train_tensor, (label_tensors[n])[ : int(instances * train_split)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    validation_set = torch.utils.data.TensorDataset(validation_tensor, (label_tensors[n])[int(instances * train_split) : int(instances * (train_split + validation_split))])
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=validation_tensor.size()[0], shuffle = False)
    test_set = torch.utils.data.TensorDataset(test_tensor, (label_tensors[n])[int(instances * (train_split + validation_split)) : ])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_tensor.size()[0], shuffle = False)

    #Initialising the CNN and gradient descender (optimizer)
    network = Network()
    if use_cuda == True:
        network = network.cuda()
    optimizer = optim.SGD(network.parameters(), lr = 0.01)

    #running the training loop
    epoch_correct : int = 0
    epoch_loss : float = 0
    lowest_valid_loss : float = float('inf')
    epochs_without_improvement = 0
    best_network = copy.deepcopy(network)

    print("results for", os.listdir(target_folder_path)[n])
    for epoch in range(0, max_training_epochs):
        previous_epoch_loss = epoch_loss
        epoch_correct = 0
        epoch_loss = 0

        for images, labels in train_loader:
            #Getting predictions before any training on this batch has occurred
            predictions = network(images)
            loss = F.cross_entropy(predictions, labels)

            #making the gradient step for this batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_correct += predictions.argmax(dim=1).eq(labels).int().sum().item() 
            epoch_loss += loss.item()


        valid_preds = network(validation_tensor)
        valid_loss = F.cross_entropy(valid_preds, label_tensors[n][int(instances * train_split) : int(instances * (train_split + validation_split))])

        if (lowest_valid_loss > valid_loss) :
            lowest_valid_loss = valid_loss
            best_network = copy.deepcopy(network)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epochs_without_improvement > 10) :
            print("stopping early")
            break

        print("epoch: ", epoch, "\ttrain_loss: ", round(epoch_loss, 5), "\ttrain_correct: ", epoch_correct, "\tvalidation_loss: ", round(valid_loss.item(),5), sep='' )
    
    test_preds = best_network(test_tensor)
    test_loss = F.cross_entropy(test_preds, label_tensors[n][int(instances * (train_split + validation_split)) : ])
    test_correct = test_preds.argmax(dim=1).eq(label_tensors[n][int(instances * (train_split + validation_split)) : ]).int().sum().item() 
    print("test_correct: ", test_correct, "/", test_preds.size()[0],  "\ttest_loss: ", round(test_loss.item(), 5), sep='' )
