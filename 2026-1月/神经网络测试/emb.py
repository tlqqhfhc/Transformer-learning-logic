import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_blobs

max_len = 15
char_list = ['*', '0', '1', '~', '|', '&', '(' , ')']

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
state_dict = torch.load('2025-12月/神经网络训练/子串法,Transformer,maxlen=15,d_m=16,d_qkv=16,L=6,hidden_size1=32,hiddensize2=16,x=0.8,epoch=200,模型结果/子串法,Transformer,maxlen=15,d_m=16,d_qkv=16,L=6,hidden_size1=32,hiddensize2=16,x=0.8,epoch=199,模型参数.pth', weights_only=True)
model.load_state_dict(state_dict)

def PCA_(data):
    data = data.numpy().T
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

with torch.no_grad():
    emb = model.embedding.weight.cpu().numpy()
    emb_tensor = torch.tensor(emb, dtype=torch.float32)
    reduced_emb = PCA_(emb_tensor)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_emb[:, 0], y=reduced_emb[:, 1])
    for i, char in enumerate(char_list):
        plt.text(reduced_emb[i, 0]+0.01, reduced_emb[i, 1]+0.01, char, fontsize=12)
    plt.title('PCA of Character Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig('2026-1月/神经网络测试/Transformer字符嵌入PCA降维可视化.png')
    plt.show()