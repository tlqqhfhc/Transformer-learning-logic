import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import random_split, Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('2025-12月/数据生成-子串法/子串法,maxlen=15,m=5,size=10^5,train.csv')
test_data = pd.read_csv('2025-12月/数据生成-子串法/子串法,maxlen=15,m=5,size=10^5,test.csv')

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

#data = train_seq + test_seq
#ans = list(train_data['ans'].values) + list(test_data['ans'].values)
#awa = range(len(data))
#np.random.shuffle(list(awa))
#train_seq = [data[i] for i in awa[:len(train_seq)]]
#test_seq = [data[i] for i in awa[len(train_seq):]]
#test_values = [ans[i] for i in awa[len(train_seq):]]
#train_values = [ans[i] for i in awa[:len(train_seq)]]

train_expr = torch.tensor([[one_hot(char_list.index(char)) for char in train_seq[i]] for i in range(len(train_seq))])
train_label = torch.tensor(train_data['ans'].values).float()
#train_label = torch.tensor(train_values).float()
test_expr = torch.tensor([[one_hot(char_list.index(char)) for char in test_seq[i]] for i in range(len(test_seq))])
test_label = torch.tensor(test_data['ans'].values).float()
#test_label = torch.tensor(test_values).float()

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

class Transformer(nn.Module):
    def __init__(self, n, d_m, d_qkv, L, hidden_size1, hidden_size2):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(len(char_list), d_m)
        self.positional_encoding = nn.Parameter(torch.zeros(1, n, d_m))
        self.layernorm = nn.LayerNorm(d_m)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.d_qkv = d_qkv
        self.W_q = nn.ModuleList([nn.Linear(d_m, d_qkv) for _ in range(L)])
        self.W_k = nn.ModuleList([nn.Linear(d_m, d_qkv) for _ in range(L)])
        self.W_v = nn.ModuleList([nn.Linear(d_m, d_qkv) for _ in range(L)])
        self.W_o = nn.ModuleList([nn.Linear(d_qkv, d_m) for _ in range(L)])
        self.fc1 = nn.ModuleList([nn.Linear(d_m, hidden_size1) for _ in range(L)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size1, hidden_size2) for _ in range(L)])
        self.fc3 = nn.ModuleList([nn.Linear(hidden_size2, d_m) for _ in range(L)])
        self.proj = nn.Linear(d_m, 1)
        def mask(scores):
            mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1))).to(scores.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            return scores
        self.mask = mask
    
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        for _ in range(len(self.W_q)):
            x_ = x
            Q = self.W_q[_](x)
            K = self.W_k[_](x)
            V = self.W_v[_](x)
            scores = self.mask(torch.matmul(Q, K.transpose(-2, -1))) / (self.d_qkv ** 0.5)
            attn_weights = self.softmax(scores)
            attn_output = torch.matmul(attn_weights, V)
            x = self.W_o[_](attn_output)
            x = self.layernorm(x + x_)
            x_ = x
            x = self.fc1[_](x)
            x = self.relu(x)
            x = self.fc2[_](x)
            x = self.relu(x)
            x = self.fc3[_](x)
            x = self.relu(x)
            x = self.layernorm(x + x_)
        x = self.proj(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1)
        x = x[:, -1]
        return x

def init(n, d_m, d_qkv, L, hidden_size1, hiddensize2, x):
    model = Transformer(n, d_m, d_qkv, L, hidden_size1, hiddensize2)
    normal_std = 2 * ((1 / d_m) ** x)
    uniform_limit = np.sqrt(2) * np.sqrt((3 / d_m) ** x)
    model.embedding.weight.data.normal_(0, normal_std)
    model.embedding.bias.data.uniform_(-uniform_limit, uniform_limit)
    model.positional_encoding.data.normal_(0, normal_std)
    for i in range(L):
        model.W_q[i].weight.data.normal_(0, normal_std)
        model.W_q[i].bias.data.uniform_(-uniform_limit, uniform_limit)
        model.W_k[i].weight.data.normal_(0, normal_std)
        model.W_k[i].bias.data.uniform_(-uniform_limit, uniform_limit)
        model.W_v[i].weight.data.normal_(0, normal_std)
        model.W_v[i].bias.data.uniform_(-uniform_limit, uniform_limit)
        model.W_o[i].weight.data.normal_(0, normal_std)
        model.W_o[i].bias.data.uniform_(-uniform_limit, uniform_limit)
        model.fc1[i].weight.data.normal_(0, normal_std)
        model.fc1[i].bias.data.uniform_(-uniform_limit, uniform_limit)
        model.fc2[i].weight.data.normal_(0, normal_std)
        model.fc2[i].bias.data.uniform_(-uniform_limit, uniform_limit)
        model.fc3[i].weight.data.normal_(0, normal_std)
        model.fc3[i].bias.data.uniform_(-uniform_limit, uniform_limit)
    model.proj.weight.data.normal_(0, normal_std)
    model.proj.bias.data.uniform_(-uniform_limit, uniform_limit)
    return model

d_m=16
d_qkv=16
L=6
hidden_size1 = 32
hiddensize2 = 16
x = 0.8
model = init(n=max_len, d_m = d_m, d_qkv = d_qkv, L = L, hidden_size1 = hidden_size1, hiddensize2 = hiddensize2, x = x)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss = []
train_accuracy = []
test_accuracy = []

num_epochs = 200
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
        torch.save(model.state_dict(), f'2025-12月/神经网络训练/子串法,Transformer,maxlen={max_len},d_m={d_m},d_qkv={d_qkv},L={L},hidden_size1={hidden_size1},hiddensize2={hiddensize2},x={x},epoch={epoch},模型参数.pth')

plt.xticks(range(1, num_epochs + 1))
plt.plot(range(1, num_epochs + 1), train_accuracy, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.savefig(f'2025-12月/神经网络训练/子串法,Transformer,maxlen={max_len},d_m={d_m},d_qkv={d_qkv},L={L},hidden_size1={hidden_size1},hiddensize2={hiddensize2},x={x},准确率曲线.png')
