import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt

# -------------------------
# 数据生成（训练长度 ≤50，测试长度可达100）
# -------------------------
TOKENS = ['a', 'b', 'c', 'd', 'e']
TOKEN2IDX = {tok: i for i, tok in enumerate(TOKENS)}
MAX_SEQ_LEN = 100

def generate_example(max_len=50):
    length = random.randint(10, max_len)
    seq = [random.choice(TOKENS) for _ in range(length)]
    count_a = seq.count('a')
    count_b = seq.count('b')
    label = 1 if count_a > count_b else 0
    seq_idx = [TOKEN2IDX[t] for t in seq]
    return seq_idx, label

class CountingDataset(Dataset):
    def __init__(self, size=5000, max_len=50):
        self.data = [generate_example(max_len) for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        # padding to MAX_SEQ_LEN
        seq = seq + [4]*(MAX_SEQ_LEN - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# -------------------------
# NoPE Transformer 模型（无位置编码）
# -------------------------
class NoPE_Transformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # 不加位置编码
        x = x.permute(1, 0, 2)  # transformer 需要 [seq_len, batch, d_model]
        x = self.transformer(x)
        x = x.mean(dim=0)  # 对序列取平均
        x = self.fc(x)
        return self.sigmoid(x)

# -------------------------
# 训练和测试集
# -------------------------
train_dataset = CountingDataset(size=5000, max_len=100)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试集包含训练长度内和长度泛化（51~100）
test_dataset_short = CountingDataset(size=500, max_len=50)
test_dataset_long = CountingDataset(size=500, max_len=100)
test_loader_short = DataLoader(test_dataset_short, batch_size=64)
test_loader_long = DataLoader(test_dataset_long, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NoPE_Transformer().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

awa = []
# -------------------------
# 训练循环
# -------------------------
for epoch in range(10):
    model.train()
    total_loss = 0
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(seq).squeeze()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataset):.4f}")
    awa.append(total_loss / len(train_dataset))

# -------------------------
# 测试函数
# -------------------------
def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for seq, label in loader:
            seq, label = seq.to(device), label.to(device)
            output = model(seq).squeeze()
            pred = (output > 0.5).float()
            correct += (pred == label).sum().item()
    return correct / len(loader.dataset)

acc_short = evaluate(test_loader_short)
acc_long = evaluate(test_loader_long)
print(f"Test Accuracy (training length ≤50): {acc_short:.4f}")
print(f"Test Accuracy (length generalization 51~100): {acc_long:.4f}")

plt.plot(awa)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss over Epochs')
plt.show()