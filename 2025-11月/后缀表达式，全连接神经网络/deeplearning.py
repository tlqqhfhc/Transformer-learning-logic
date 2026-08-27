import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split, Dataset, DataLoader

data = pd.read_csv('data.csv')
#data = data[:1000]

awa = [list(data['expr'].values[i]) for i in range(len(data))]
max_len = max([len(awa[i]) for i in range(len(awa))])
print(f'Max expression length in dataset: {max_len}')
for i in range(len(awa)):
    while len(awa[i]) < max_len:
        awa[i] = awa[i] + ['*']

char_list = ['*', '1', '~', '|', '&']
def one_hot(x):
    return [1.0 if i == x else 0.0 for i in range(len(char_list))]

expr = torch.tensor([[one_hot(char_list.index(char)) for char in awa[i]] for i in range(len(awa))])
label = torch.tensor(data['label'].values).float()

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

full_data = CustomDataset(expr, label)
train_size = int(0.7 * len(data))
train_data, check_data = random_split(full_data, [train_size, len(data) - train_size])

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
check_loader = DataLoader(check_data, batch_size=batch_size, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(NeuralNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

model = NeuralNetwork(input_size=max_len*len(char_list), hidden_size1=128, hidden_size2=64)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    tot_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {(tot_loss / len(train_loader)):.4f}')
    if epoch % 10 == 9:
        with torch.no_grad():
            tot_loss = 0
            tot = 0
            for i, (inputs, targets) in enumerate(check_loader):
                outputs = model.forward(inputs) 
                loss = criterion(outputs.squeeze(), targets)
                tot_loss += loss.item()
                tot += ((outputs.squeeze() >= 0.5).float() == targets ).sum().item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Check_Loss: {(tot_loss / len(check_loader)):.4f}, Check_Accuracy: {tot / (len(check_loader) * batch_size):.4f}')
        torch.save(model, f'model_epoch{epoch+1}.pth')
