import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split, Dataset, DataLoader
import numpy as np

train_data = pd.read_csv('2025-12月/数据生成-子串法/train.csv')
test_data = pd.read_csv('2025-12月/数据生成-子串法/test.csv')
print('Data loaded successfully.')

train_seq = [list(train_data['seq'].values[i]) for i in range(len(train_data))]
test_seq = [list(test_data['seq'].values[i]) for i in range(len(test_data))]
max_len = max(max([len(train_seq[i]) for i in range(len(train_seq))]), max([len(test_seq[i]) for i in range(len(test_seq))]))

for i in range(len(train_seq)):
    while len(train_seq[i]) < max_len:
        train_seq[i] = train_seq[i] + ['*']
for i in range(len(test_seq)):
    while len(test_seq[i]) < max_len:
        test_seq[i] = test_seq[i] + ['*']

char_list = ['*', '0', '1', '~', '|', '&', '(' , ')']
def one_hot(x):
    return [1.0 if i == x else 0.0 for i in range(len(char_list))]

train_expr = torch.tensor([[one_hot(char_list.index(char)) for char in train_seq[i]] for i in range(len(train_seq))])
train_label = torch.tensor(train_data['ans'].values).float()
test_expr = torch.tensor([[one_hot(char_list.index(char)) for char in test_seq[i]] for i in range(len(test_seq))])
test_label = torch.tensor(test_data['ans'].values).float()

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(train_expr, train_label)
test_dataset = CustomDataset(test_expr, test_label)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class FNN(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, hidden_size3, n):
        super(FNN, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(n * len(char_list), hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)
    
    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x.squeeze()

model = FNN(hidden_size1 = 32, hidden_size2 = 64, hidden_size3 = 32, n = max_len)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss = []
train_accuracy = []
test_accuracy = []

num_epochs = 100
for epoch in range(num_epochs):
    tot_loss = 0
    tot = 0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot += ((outputs >= 0.5).float() == targets ).sum().item()
    train_loss.append(tot_loss / len(train_loader))
    train_accuracy.append(tot / (len(train_loader) * batch_size))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss[epoch]:.4f}, Train_Accurancy: {train_accuracy[epoch]:.4f}')
    with torch.no_grad():
        tot = 0
        for i, (inputs, targets) in enumerate(test_loader):
            outputs = model.forward(inputs) 
            outputs = (outputs >= 0.5).float()
            tot += (outputs == targets).sum().item()
        test_accuracy.append(tot / (len(test_loader) * batch_size))
        print(f'Epoch [{epoch+1}/{num_epochs}], Test_Accurancy: {test_accuracy[epoch]:.4f}')
    if epoch % 10 == 9:
        torch.save(model.state_dict(), f'2025-12月/神经网络训练/FNN_epoch{epoch+1}.pth')
