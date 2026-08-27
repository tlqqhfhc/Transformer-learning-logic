import pandas as pd
import matplotlib.pyplot as plt

m = 7
df = pd.read_csv('2025-12月/神经网络测试/accuracy_results.csv')

data = df.values
averaged_data = {
    'epoch': [_ + (m + 1) / 2 for _ in range(200 - m + 1)],
    'not_only_accuracy': [data[_:_ + m, 1].mean() for _ in range(200 - m + 1)],
    'or_only_accuracy': [data[_:_ + m, 2].mean() for _ in range(200 - m + 1)],
    'and_only_accuracy': [data[_:_ + m, 3].mean() for _ in range(200 - m + 1)],
    'small_accuracy': [data[_:_ + m, 4].mean() for _ in range(200 - m + 1)],
    'test_accuracy': [data[_:_ + m, 5].mean() for _ in range(200 - m + 1)],
    'train_accuracy': [data[_:_ + m, 6].mean() for _ in range(200 - m + 1)],
}
df = pd.DataFrame(averaged_data)

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['not_only_accuracy'], label='Not Only Data Accuracy')
plt.plot(df['epoch'], df['or_only_accuracy'], label='Or Only Data Accuracy')
plt.plot(df['epoch'], df['and_only_accuracy'], label='And Only Data Accuracy')
plt.plot(df['epoch'], df['small_accuracy'], label='Small Data Accuracy')
plt.plot(df['epoch'], df['test_accuracy'], label='Test Data Accuracy')
plt.plot(df['epoch'], df['train_accuracy'], label='Train Data Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs')
plt.legend()
plt.grid()
plt.savefig(f'2025-12月/神经网络测试/accuracy_plot,averaged(m={m}).png')
plt.show()