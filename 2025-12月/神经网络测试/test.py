import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

max_len = 15
not_only_data = pd.read_csv('2025-12月/数据生成-特殊数据/仅非运算,maxlen=15,size=10^2.csv')
or_only_data = pd.read_csv('2025-12月/数据生成-特殊数据/仅或运算,maxlen=15,size=10^2.csv')
and_only_data = pd.read_csv('2025-12月/数据生成-特殊数据/仅与运算,maxlen=15,size=10^2.csv')
small_data = pd.read_csv('2025-12月/数据生成-特殊数据/小规模随机表达式,maxlen=10,size=10^2.csv')
test_data = pd.read_csv('2025-12月/数据生成-子串法/子串法,maxlen=15,m=5,size=10^5,test.csv')
train_data = pd.read_csv('2025-12月/数据生成-子串法/子串法,maxlen=15,m=5,size=10^5,train.csv')
train_data = train_data.sample(frac=0.1, random_state=7845).reset_index(drop=True)

char_list = ['*', '0', '1', '~', '|', '&', '(' , ')']
def one_hot(x):
    return [1.0 if i == x else 0.0 for i in range(len(char_list))]

def preprocess(seq: str):
    while len(seq) < max_len:
        seq = seq + '*'
    one_hot_seq = [one_hot(char_list.index(c)) for c in seq]
    return one_hot_seq

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

not_only_dataset = CustomDataset(torch.tensor([preprocess(not_only_data['seq'].values[i]) for i in range(len(not_only_data))]), torch.tensor(not_only_data['ans'].values).float())
or_only_dataset = CustomDataset(torch.tensor([preprocess(or_only_data['seq'].values[i]) for i in range(len(or_only_data))]), torch.tensor(or_only_data['ans'].values).float())
and_only_dataset = CustomDataset(torch.tensor([preprocess(and_only_data['seq'].values[i]) for i in range(len(and_only_data))]), torch.tensor(and_only_data['ans'].values).float())
small_dataset = CustomDataset(torch.tensor([preprocess(small_data['seq'].values[i]) for i in range(len(small_data))]), torch.tensor(small_data['ans'].values).float())
test_dataset = CustomDataset(torch.tensor([preprocess(test_data['seq'].values[i]) for i in range(len(test_data))]), torch.tensor(test_data['ans'].values).float())
train_dataset = CustomDataset(torch.tensor([preprocess(train_data['seq'].values[i]) for i in range(len(train_data))]), torch.tensor(train_data['ans'].values).float())

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

model = Transformer(n=max_len, d_m=16, d_qkv=16, L=6, hidden_size1=32, hidden_size2=16)

def evaluate_model(dataset):
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        tot = 0
        with torch.no_grad():
            tot = 0
            for i, (inputs, targets) in enumerate(loader):
                outputs = model.forward(inputs)
                outputs = (outputs >= 0.5).float()
                tot += (outputs == targets).sum().item()
        accuracy = tot / len(dataset)
        return accuracy

not_only_accuracy = []
or_only_accuracy = []
and_only_accuracy = []
small_accuracy = []
test_accuracy = []
train_accuracy = []

for epoch in range(200):
    state_dict = torch.load(f'2025-12月/神经网络训练/子串法,Transformer,maxlen=15,d_m=16,d_qkv=16,L=6,hidden_size1=32,hiddensize2=16,x=0.8,epoch=200,模型结果/子串法,Transformer,maxlen=15,d_m=16,d_qkv=16,L=6,hidden_size1=32,hiddensize2=16,x=0.8,epoch={epoch},模型参数.pth', weights_only=True)
    model.load_state_dict(state_dict)

    not_only_accuracy.append(evaluate_model(not_only_dataset))
    or_only_accuracy.append(evaluate_model(or_only_dataset))
    and_only_accuracy.append(evaluate_model(and_only_dataset))
    small_accuracy.append(evaluate_model(small_dataset))
    test_accuracy.append(evaluate_model(test_dataset))
    train_accuracy.append(evaluate_model(train_dataset))
    print(f'Epoch {epoch + 1}:')
    print(f'    not only data accurancy: {not_only_accuracy[-1]:.4f}')
    print(f'    or only data accurancy: {or_only_accuracy[-1]:.4f}')
    print(f'    and only data accurancy: {and_only_accuracy[-1]:.4f}')
    print(f'    small data accurancy: {small_accuracy[-1]:.4f}')
    print(f'    test data accurancy: {test_accuracy[-1]:.4f}')
    print(f'    train data accurancy: {train_accuracy[-1]:.4f}')

save_dict = {
    'epoch': list(range(1, 201)),
    'not_only_accuracy': not_only_accuracy,
    'or_only_accuracy': or_only_accuracy,
    'and_only_accuracy': and_only_accuracy,
    'small_accuracy': small_accuracy,
    'test_accuracy': test_accuracy,
    'train_accuracy': train_accuracy,
}
df = pd.DataFrame(save_dict)
df.to_csv('2025-12月/神经网络测试/accuracy_results.csv', index=False)