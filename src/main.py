import pandas as pd
# import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import time
import matplotlib.pyplot as plt
import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data = pd.read_csv("C:/Users/kjpoo/Documents/Git_projects/Data_Science_Projects/data/data.csv")
data = pd.read_csv("../data/data.csv")
BATCH_SIZE = 512

# Split the data into train, validation and test.
train_all_data, test_data = train_test_split(data, test_size=0.3)
train_data, validation_data = train_test_split(train_all_data, test_size=0.3)

print("Size of Train data: ", len(train_data))
print("Size of Validation data: ", len(validation_data))
print("Size of Test data: ", len(test_data))

# Plot the distribution of the sample size of each number.
x = data['label'].value_counts().values
y = data['label'].value_counts().index
sns.barplot(y,x)
plt.show()

X_train_all = train_all_data.iloc[:,1:].astype("float32")
Y_train_all = train_all_data.iloc[:,0].astype("int32")

X_train = train_data.iloc[:,1:].astype("float32")
Y_train = train_data.iloc[:,0].astype("int32")

X_val = validation_data.iloc[:,1:].astype("float32")
Y_val = validation_data.iloc[:,0].astype("int32")

X_test = test_data.iloc[:,1:].astype("float32")
Y_test = test_data.iloc[:,0].astype("int32")

# Convert to tensor
X_train_all_tensor = torch.tensor(X_train_all.values).reshape(len(X_train_all),1,28,28)
Y_train_all_tensor = torch.tensor(Y_train_all.values)

X_train_tensor = torch.tensor(X_train.values).reshape(len(X_train),1,28,28)
Y_train_tensor = torch.tensor(Y_train.values)

X_val_tensor = torch.tensor(X_val.values).reshape(len(X_val),1,28,28)
Y_val_tensor = torch.tensor(Y_val.values)

X_test_tensor = torch.tensor(X_test.values).reshape(len(X_test),1,28,28)
Y_test_tensor = torch.tensor(Y_test.values)

for i in range(5):
    print("Label:",Y_train_tensor[i].item())
    plt.imshow(X_train_tensor[i].reshape(28,28,1),cmap="gray") 
    plt.show()


class trainDataset(Dataset):
    def __init__(self, X, Y):
        self.Xdata = X
        self.Ydata = Y
    def __len__(self):
        return len(X_train_tensor)
    def __getitem__(self, index):
        result = (self.Xdata[index], self.Ydata[index])
        return result

train_dataloader = DataLoader(trainDataset(X_train_tensor, Y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
train_all_dataloader = DataLoader(trainDataset(X_train_all_tensor, Y_train_all_tensor), batch_size=BATCH_SIZE, shuffle=True)

class Model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(4 * 4 * 128, 625, bias=True),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(625, out_ch, bias=True),
            nn.Softmax()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

model = Model(1,10)
print(model)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.009)

train_cost = []
train_accu = []
val_accu = []
training_epochs = 30

start_time = time.time()
print("start time: ", start_time)

for epoch in range(training_epochs):
    model.to(device)
    model.train()
    accuracy = 0
    cost = 0.0
    
    for idx, (batch_X, batch_Y) in enumerate(train_dataloader):
        batchX = batch_X.to(device) 
        batchY = batch_Y.to(device)
        predicted_output = model(batchX)
        
        batchY = batchY.long()
        
        optimizer.zero_grad()
        loss = criterion(predicted_output, batchY)
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(predicted_output.data, 1)
        accuracy += (pred == batchY).sum().item()
        
        cost += loss.item()
        
    train_cost.append(cost)
    train_accu.append(accuracy/len(X_train_tensor))
    
    val_accuracy = 0
    
    model.to('cpu')
    model.eval()
    val_pred_output = model(X_val_tensor)
    
    _, pred = torch.max(val_pred_output.data, 1)
    val_accuracy += (pred == Y_val_tensor).sum().item()
    
    val_accu.append(val_accuracy/len(validation_data))
    
    print("epoch:", epoch, "| Train cost:", cost, "| Train Accuracy:", train_accu[-1], "| validation Accuracy:", val_accu[-1])


end_time = time.time()
print("end time: ", end_time)
print("Total time: ", end_time - start_time)

for i in range(5):
    print("Label:", pred[i].item())
    plt.imshow(X_val_tensor[i].reshape(28,28,1),cmap="gray") 
    plt.show()    

plt.plot(val_accu, label="validation")
plt.plot(train_accu, label="Training")
plt.legend()
plt.show()

start_time = time.time()
print("start time: ", start_time)

training_epochs = 30
model_final = Model(1,10).to(device)
optimizer = torch.optim.Adam(params=model_final.parameters(), lr=learning_rate, weight_decay=0)

for epoch in range(training_epochs):
    model.train()
    cost = 0.0
    for idx, (batch_X, batch_Y) in enumerate(train_all_dataloader):
        batchX = batch_X.to(device) 
        batchY = batch_Y.to(device)
        predicted_output = model_final(batchX)
        batchY = batchY.long()
        optimizer.zero_grad()
        loss = criterion(predicted_output, batchY)
        loss.backward()
        optimizer.step()
        cost += loss.item()
    print("epoch:", epoch,"| cost:", cost)

model_final.to('cpu')
model_final.eval()

test_accuracy = 0

test_pred_output = model_final(X_test_tensor)
_, pred = torch.max(test_pred_output.data, 1)

test_accuracy = ((pred == Y_test_tensor).sum().item())/len(test_data)

print("\nTest accuracy:", test_accuracy)
print("CONFUSION MATRIX:\n", multilabel_confusion_matrix(pred,Y_test_tensor))
print("Classification Report:\n", classification_report(pred,Y_test_tensor))