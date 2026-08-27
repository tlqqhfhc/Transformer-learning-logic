import torch
import copy
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn

TOKENS = ['a', 'b', 'c', 'd', 'e']
TOKEN2IDX = {tok: i for i, tok in enumerate(TOKENS)}
MAX_SEQ_LEN = 100

class ParityDataset(Dataset):
    def __init__(self, size=5000, max_len=100):
        self.data = [generate_parity_example(max_len) for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        seq = seq + [0]*(MAX_SEQ_LEN - len(seq))  # padding
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def generate_parity_example(max_len=100):
    length = random.randint(10, max_len)
    seq = [random.choice(TOKENS) for _ in range(length)]
    count_b = seq.count('b')
    label = 1 if count_b % 2 == 1 else 0  # 1 = 奇数，0 = 偶数
    seq_idx = [TOKEN2IDX[t] for t in seq]
    return seq_idx, label

test_dataset = ParityDataset(size=500, max_len=50)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def load_model(model_class, path, device='cpu'):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def replace_attention(target_model, source_model):
    for t_layer, s_layer in zip(target_model.transformer.layers,
                                source_model.transformer.layers):
        t_layer.self_attn.load_state_dict(s_layer.self_attn.state_dict())

def replace_ffn(target_model, source_model):
    for t_layer, s_layer in zip(target_model.transformer.layers,
                                source_model.transformer.layers):
        t_layer.linear1.load_state_dict(s_layer.linear1.state_dict())
        t_layer.linear2.load_state_dict(s_layer.linear2.state_dict())

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            total += y.size(0)
            out = model(x).squeeze()
            pred = (out > 0.5).float()
            correct += (pred == y).sum().item()
    return correct / total


# -------- 使用 --------
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_before = load_model(NoPE_Transformer, model_before_path, device)
model_after = load_model(NoPE_Transformer, model_after_path, device)

acc_before = evaluate(model_before, test_loader, device)
acc_after = evaluate(model_after, test_loader, device)
# ✅ 实验1：替换 attention
model_mix_attn = copy.deepcopy(model_after)
replace_attention(model_mix_attn, model_before)

# ✅ 实验2：替换 FFN
model_mix_ffn = copy.deepcopy(model_after)
replace_ffn(model_mix_ffn, model_before)

# 测试
acc_mix_attn = evaluate(model_mix_attn, test_loader, device)
acc_mix_ffn = evaluate(model_mix_ffn, test_loader, device)

print("Before:", acc_before)
print("After:", acc_after)
print("Replace Attention:", acc_mix_attn)
print("Replace FFN:", acc_mix_ffn)