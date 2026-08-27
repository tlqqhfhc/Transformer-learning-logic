import numpy as np
from typing import List, Tuple
import hashlib
import pandas as pd

Max_Length = 15
m = 5
size = 200000

def split(seq: str) -> bool:
    """
    属于训练集则返回0
    否则返回1
    """
    hash_object = hashlib.md5(seq.encode())
    return int(hash_object.hexdigest(), 16) % 3 == 0

char_list = ['1', '0', '|', '~', '&', '(', ')']
next_char_list = {
    '1': ['|', '&', ')'],
    '0': ['|', '&', ')'],
    '|': ['1', '0', '(', '~'],
    '~': ['1', '0', '(', '~'],
    '&': ['1', '0', '(', '~'],
    '(': ['1', '0', '(', '~'],
    ')': ['|', '&', ')']
}

def generate_data(seq: str, flag: bool) -> List[str]:
    """
    生成数据
    """
    if len(seq) >= Max_Length:
        if (seq.count('(') != seq.count(')')) | (seq[-1] in ['(', '|', '&', '~']):
            return []
        return [seq]
    ans = []
    for next_char in np.random.permutation(next_char_list[seq[-1]]):
        if next_char == ')' and seq.count('(') <= seq.count(')'):
            continue
        if split(seq[-4:] + next_char) == flag:
            temp = generate_data(seq + next_char, flag)
            ans = ans + temp
            if len(ans):
                break
    return ans

def calc(seq: str) -> bool:
    """
    计算逻辑表达式的值
    """
    if seq == '1':
        return 1
    if seq == '0':
        return 0
    cnt = 0
    first_and = -1
    first_or = -1
    for i in range(len(seq)):
        if first_and != -1 and first_or != -1:
            break
        if seq[i] == '(':
            cnt += 1
        elif seq[i] == ')':
            cnt -= 1
        elif cnt == 0:
            if seq[i] == '&' and first_and == -1:
                first_and = i
            if seq[i] == '|' and first_or == -1:
                first_or = i
    if first_or != -1:
        return calc(seq[:first_or]) | calc(seq[first_or + 1:])
    if first_and != -1:
        return calc(seq[:first_and]) & calc(seq[first_and + 1:])
    if seq[0] == '~':
        return 1 - calc(seq[1:])
    return calc(seq[1:-1])

Train_Data = []
Test_Data = []

for i in range(size):
    start = np.random.choice(['1', '0', '(', '~'])
    for _ in range(m - 1):
        nxt = ')'
        while nxt.count('(') < nxt.count(')'):
            nxt = start + np.random.choice(next_char_list[start[-1]])
        start = nxt
    data = generate_data(start, split(start))
    if len(data) == 0:
        continue
    if split(data[0][-5:]) == 0:
        Train_Data = Train_Data + data
    else:
        Test_Data = Test_Data + data
    if int(i / (size // 10)) != int((i - 1) / (size // 10)):
        print(f'Progress: {i}/{size}')
np.random.shuffle(Train_Data)
np.random.shuffle(Test_Data)

Train_data = [[item, calc(item)] for item in Train_Data]
Test_data = [[item, calc(item)] for item in Test_Data]

pd.DataFrame(Train_data, columns = ['seq', 'ans']).to_csv(f'2025-12月/数据生成-子串法/子串法,maxlen={Max_Length},m={m},size=10^{int(np.log10(size))},train.csv', index = False)
pd.DataFrame(Test_data, columns = ['seq', 'ans']).to_csv(f'2025-12月/数据生成-子串法/子串法,maxlen={Max_Length},m={m},size=10^{int(np.log10(size))},test.csv', index = False)
if __name__ == "__main__":
    print("Train and Test data generated and saved to CSV files.")
    print("Train data size:", len(Train_data))
    print("Test data size:", len(Test_data))