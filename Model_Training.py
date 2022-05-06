import dataprocess as dp
import transformers
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from modelhelper import MSE_Vec_matrix
from modelhelper import Net
import torch
from torchvision import models


#TODO Normalize Xtest
#Make our own loss function

# Hyperparameters
# Batch Size, Num Epochs, Learning Rate, Momentum, FC Layer, Activation Function


# 0 Lowercase & N2W
# 1 Contractions
# 2 Remove Punctutations
# 3 Strop Words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(net, input_id, labels):
    j = len(labels)

    guesses = torch.argsort(net(input_id), dim=1, descending=True)
#     print("GUESS: ", guesses[:,0])

    # guess1 = torch.argmax(guesses, dim=1)
    # guess2 = torch.argsort(guesses, dim=1)[-2]


    # current_real = torch.tensor(labels)
    current_real = labels.clone().detach()
#     print("LABEL: ", current_real)

    top1_acc = torch.sum(current_real==guesses[:,0])
    top2_acc = torch.sum(current_real==guesses[:,1])


    # print(guesses[:100])
    # print(current_real[:100])
    # print("Guess: ", guesses, "Label: ", current_real)

    accuracy_1 = top1_acc / j
    accuracy_2 = (top2_acc + top1_acc) / j
    # print(running_acc)
    # print(j)

    # print('Accuracy for top 1: %d %%' % ((accuracy_1) * 100.0))
    # print('Accuracy for top 2: %d %%' % ((accuracy_2) * 100.0))

    return accuracy_1.item(), accuracy_2.item()



data = dp.unpickle_data("reviews_Electronics_5_7_encoded.pickle")



sentences = data['reviewText']
labels = data['overall']
print(sentences.shape)
print(len(labels))


X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=42, stratify=labels)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

X_test = X_test.clone().detach()
y_test = torch.tensor(y_test).cuda()

for i in range(len(y_test)):
    y_test[i] = y_test[i] - 1 


# indecies = torch.permute(indecies)
# X_train = X_train[indexes,:]

X_train = X_train.clone().detach()
y_train = torch.tensor(y_train)



rev1 = X_train[y_train==1]
rev2 = X_train[y_train==2]
rev3 = X_train[y_train==3]
rev4 = X_train[y_train==4]
rev5 = X_train[y_train==5]

training_points = min(len(rev1), len(rev2), len(rev3), len(rev4), len(rev5))

rev1 = rev1 [:training_points]
rev2 = rev2 [:training_points]
rev3 = rev3 [:training_points]
rev4 = rev4 [:training_points]
rev5 = rev5 [:training_points]

indicies = torch.randperm(training_points * 5)

X_train = torch.cat((rev1, rev2, rev3, rev4, rev5), dim=0)[indicies, :]
base = torch.zeros(training_points)
y_train = torch.cat((base,base+1,base+2,base+3,base+4))[indicies]




NUM_EPOCH = 10000
batch_size = 250

# Learning Rate Decay
LR_START  = 0.3
LR_END    = 1e-3
LR_GAMMA  = (LR_END/LR_START)**(1/NUM_EPOCH)

dropout = 0.1
hidden_layers = [[768, 200], [200, 70], [70, 20], [20, 5]]
activation_func = F.sigmoid

batch_norm = True

net = Net(  h_sizes=hidden_layers,
            dropout=dropout, 
            activation=activation_func,
            batch_norm=batch_norm).to(device)

print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=LR_START, momentum=0.9)
criterion = torch.nn.MSELoss().to(device)
# criterion = torch.nn.CrossEntropyLoss().to(device)
scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
MSE_vec   = MSE_Vec_matrix(mid=0.8)
MSE_vec.to(device)




X_train = X_train.to(device)
y_train = y_train.long().to(device)
X_test  = X_test.to(device)


train_losses = []
test_losses = []
accuracy1 = []
accuracy2 = []



indecies = torch.tensor(range(batch_size))

for epoch in range(NUM_EPOCH):
    running_loss = 0.0

    # Training mode
    net.train()
    for i in range(len(y_train)//batch_size):
    #for i in range(1):
        select = batch_size * i + indecies
        optimizer.zero_grad()
        outputs = net(X_train[select,:])

        # MSE Loss
        loss = criterion(outputs, MSE_vec[y_train[select]])

        # Cross Entropy Loss
        # loss = criterion(outputs, y_train[select])

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()

    # Eval Mode
    net.eval()
    with torch.no_grad():

        # MSE
        training_loss = criterion(net(X_test),MSE_vec[y_test.long()]).item()

        # Cross Entropy
        # training_loss = criterion(net(X_test), y_test.long()).item()

        # Added .eval() to account for the added dropout layers
        ac1,ac2 = accuracy(net, X_test, y_test)
        accuracy1.append(ac1)
        accuracy2.append(ac2)
    
    print('[%d] loss: %.3f \t test loss: %.3f \t test_ac1: %.2f \t test_ac2: %.2f' %
    (epoch + 1, running_loss, training_loss*10,ac1,ac2))
    scheduler.step()
    
    train_losses.append(running_loss)
    test_losses.append(training_loss)
    running_loss = 0.0



print('Finished Training')
plt.plot(train_losses)
plt.figure()
plt.plot(test_losses)
plt.figure()
plt.plot(accuracy1)
plt.figure()
plt.plot(accuracy2)



net = net.eval()

train_acc = accuracy(net, X_train.cuda(), y_train.cuda())
test_acc = accuracy(net, X_test, y_test)

print("Train Top 1: ", round(train_acc[0], 2), "Train Top 2: ", round(train_acc[1], 2))
print("Test Top 1: ", round(test_acc[0], 2), "Test Top 2: ", round(test_acc[1], 2))
