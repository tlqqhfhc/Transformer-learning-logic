import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# -------------------------
# 基本设置
# -------------------------
TOKENS = ['a', 'b', 'c', 'd', 'e']
PAD_TOKEN = '<pad>'

TOKEN2IDX = {tok: i for i, tok in enumerate(TOKENS)}
PAD_IDX = len(TOKEN2IDX)
TOKEN2IDX[PAD_TOKEN] = PAD_IDX

VOCAB_SIZE = len(TOKEN2IDX)
MAX_SEQ_LEN = 100

# -------------------------
# 数据生成（奇偶任务）
# -------------------------
def generate_example(min_len, max_len=50):
    length = random.randint(min_len, max_len)
    seq = [random.choice(TOKENS) for _ in range(length)]
    
    count_a = seq.count('a')
    label = 1 if count_a % 2 == 1 else 0  # 奇数=1，偶数=0
    
    seq_idx = [TOKEN2IDX[t] for t in seq]
    return seq_idx, label, length

class ParityDataset(Dataset):
    def __init__(self, size=5000, min_len=5, max_len=50):
        self.data = [generate_example(min_len, max_len) for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, label, length = self.data[idx]
        
        padding_len = MAX_SEQ_LEN - len(seq)
        seq = seq + [PAD_IDX] * padding_len
        
        # attention mask（1=有效token，0=padding）
        mask = [1]*length + [0]*padding_len
        
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(label, dtype=torch.float)
        )

# -------------------------
# Transformer 模型（带mask）
# -------------------------
class ParityTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, mask):
        # x: [batch, seq_len]
        # mask: [batch, seq_len] (True=valid, False=pad)
        
        x = self.embedding(x)
        
        # PyTorch要求 padding mask: True=pad
        padding_mask = ~mask
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # 👉 关键：只对非padding位置做平均
        mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        x = x * mask
        
        x = x.sum(dim=1) / mask.sum(dim=1).float().clamp(min=1)
        
        x = self.fc(x)
        return self.sigmoid(x)

# -------------------------
# 数据加载
# -------------------------
train_dataset = ParityDataset(size=5000, min_len=5, max_len=50)
test_dataset_short = ParityDataset(size=500, min_len=5, max_len=50)
test_dataset_long = ParityDataset(size=500, min_len=51, max_len=100)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader_short = DataLoader(test_dataset_short, batch_size=32)
test_loader_long = DataLoader(test_dataset_long, batch_size=32)

# -------------------------
# 训练
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ParityTransformer(VOCAB_SIZE).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def evaluate(loader):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for seq, mask, label in loader:
            seq, mask, label = seq.to(device), mask.to(device), label.to(device)
            
            output = model(seq, mask).squeeze()
            pred = (output > 0.5).float()
            
            correct += (pred == label).sum().item()
    
    return correct / len(loader.dataset)


for epoch in range(1000):
    model.train()
    total_loss = 0
    
    for seq, mask, label in train_loader:
        seq, mask, label = seq.to(device), mask.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(seq, mask).squeeze()
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * seq.size(0)
    acc_short = evaluate(test_loader_short)
    acc_long = evaluate(test_loader_long)
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataset):.4f}")
    print(f"Test Accuracy (Short): {acc_short:.4f}")
    print(f"Test Accuracy (Long): {acc_long:.4f}")
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"awa/noPE_transformer_models/parity_transformer_epoch{epoch+1}.pth")
# -------------------------
# 测试
# -------------------------
