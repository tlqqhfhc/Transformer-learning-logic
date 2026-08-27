import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(NeuralNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
    
    def forward(self, x):#x:[32,36,5]
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

char_list = ['*', '1', '~', '|', '&']
def one_hot(x):
    return [1.0 if i == x else 0.0 for i in range(len(char_list))]
model_path = [f'model_epoch{i}.pth' for i in range(10, 101, 10)]
awa = [['1', '~', '1', '&']]
max_len = 40
for i in range(len(awa)):
    while len(awa[i]) < max_len:
        awa[i] = awa[i] + ['*']
expr = torch.tensor([[one_hot(char_list.index(char)) for char in awa[i]] for i in range(len(awa))])
for path in model_path:
    print(f'Loading model from {path}...')
    model = torch.load(path)
    model.eval()
    with torch.no_grad():
        outputs = model(expr)
        predicted = (outputs.squeeze() >= 0.5).float()
        print(f'Model: {path}, Outputs: {outputs.tolist()}, Predictions: {predicted.tolist()}')