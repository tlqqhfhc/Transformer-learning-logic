import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

TOKENS = ['a', 'b', 'c', 'd', 'e']

def load_embedding(model_class, path, device='cpu'):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.embedding.weight.data.clone().cpu()

def cosine_similarity_matrix(emb):
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.T
    return sim.numpy()

def plot_similarity(sim_matrix, title):
    plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(TOKENS)), TOKENS)
    plt.yticks(range(len(TOKENS)), TOKENS)
    plt.title(title)
    plt.show()

# -------- 使用 --------
# 替换为你的模型类
class NoPE_Transformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)  # 不加位置编码
        x = x.permute(1, 0, 2)  # transformer 需要 [seq_len, batch, d_model]
        x = self.transformer(x)
        x = x.mean(dim=0)  # 对序列取平均
        x = self.fc(x)
        return self.sigmoid(x)


model_before_path = "awa/noPE_transformer_models/noPE_transformer_epoch100.pth"
model_after_path = "awa/noPE_transformer_models/noPE_transformer_epoch980.pth"

emb_before = load_embedding(NoPE_Transformer, model_before_path)
emb_after = load_embedding(NoPE_Transformer, model_after_path)

sim_before = cosine_similarity_matrix(emb_before)
sim_after = cosine_similarity_matrix(emb_after)

plot_similarity(sim_before, "Before Grokking")
plot_similarity(sim_after, "After Grokking")