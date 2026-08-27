import pandas as pd
import numpy as np

np.random.seed(7845)
data_address = '2025-12月/数据生成-子串法/子串法,maxlen=15,m=5,size=10^5,train'
data = pd.read_csv(data_address + '.csv')

def translate(seq: str) -> str:
    if seq == '1' or seq == '0':
        return seq
    cnt = 0
    first_and = -1
    first_or = -1
    for i in range(len(seq)):
        if seq[i] == '(':
            cnt += 1
        if seq[i] == ')':
            cnt -= 1
        if cnt == 0:
            if seq[i] == '&' and first_and == -1:
                first_and = i
            if seq[i] == '|' and first_or == -1:
                first_or = i
    if first_or != -1:
        return translate(seq[:first_or]) + translate(seq[first_or + 1:]) + '|'
    if first_and != -1:
        return translate(seq[:first_and]) + translate(seq[first_and + 1:]) + '&'
    if seq[0] == '~':
        return translate(seq[1:]) + '~'
    return translate(seq[1:-1])

data['seq'] = data['seq'].apply(translate)
data.to_csv(data_address + ',后缀.csv', index = False)