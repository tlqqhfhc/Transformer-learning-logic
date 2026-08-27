import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

max_len = 15

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
            attn.append(attn_weights)
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
state_dict = torch.load('2025-12月/神经网络训练/子串法,Transformer,maxlen=15,d_m=16,d_qkv=16,L=6,hidden_size1=32,hiddensize2=16,x=0.8,epoch=200,模型结果/子串法,Transformer,maxlen=15,d_m=16,d_qkv=16,L=6,hidden_size1=32,hiddensize2=16,x=0.8,epoch=199,模型参数.pth', weights_only=True)
model.load_state_dict(state_dict)

def forward_with_attn(m, x, attention):
        x = m.embedding(x)
        x = x + m.positional_encoding
        for _ in range(len(m.W_q)):
            x_ = x
            V = m.W_v[_](x)
            attn_weights = attention[_]
            attn_output = torch.matmul(attn_weights, V)
            x = m.W_o[_](attn_output)
            x = m.layernorm(x + x_)
            x_ = x
            x = m.fc1[_](x)
            x = m.relu(x)
            x = m.fc2[_](x)
            x = m.relu(x)
            x = m.fc3[_](x)
            x = m.relu(x)
            x = m.layernorm(x + x_)
        x = m.proj(x)
        x = m.sigmoid(x)
        x = x.squeeze(-1)
        x = x[:, -1]
        return x

test_seq1 = ['((1&0)|(~0&1))', '~(1|0)&(1&0)', '~~(1|0)&(1&0)', '((1&0)|(1|~1))', '((1&0)|(0|1))']
test_seq2 = ['((1&0)&(~0&1))', '~(1|0)|(1&0)', '~~(1|0)|(1&0)', '((1&0)|(1&~1))', '((1&0)&(0|1))']

with torch.no_grad():
    for _ in range(len(test_seq1)):
        seq1 = preprocess(test_seq1[_])
        seq2 = preprocess(test_seq2[_])
        seq_tensor1 = torch.tensor([seq1], dtype=torch.float32)
        seq_tensor2 = torch.tensor([seq2], dtype=torch.float32)
        attn = []
        output1 = model(seq_tensor1)
        attn1 = attn
        attn = []
        output2 = model(seq_tensor2)
        attn2 = attn
        print(f'Sequence: {test_seq1[_]}, Output: {output1.item():.4f}')
        print(f'Sequence: {test_seq2[_]}, Output: {output2.item():.4f}')
        output1 = forward_with_attn(model, seq_tensor1, attn2)
        output2 = forward_with_attn(model, seq_tensor2, attn1)
        print(f'After swapping attention:')
        print(f'Sequence: {test_seq1[_]}, Output: {output1.item():.4f}')
        print(f'Sequence: {test_seq2[_]}, Output: {output2.item():.4f}')