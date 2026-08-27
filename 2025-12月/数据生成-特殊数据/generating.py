import pandas as pd
import numpy as np
np.random.seed(845652)
min_len = 10
max_len = 15
count = 512

def or_only(len_):
    if len_ % 2 == 0:
        len_ -= 1
    def gene(len):
        if len == 1:
            return np.random.choice(['0', '1'])
        if len == 2:
            return 'error'
        if np.random.rand() < 0.7:
            pos = np.random.randint(1, len - 1)
            left_len = pos
            right_len = len - pos - 1
            left = gene(left_len)
            right = gene(right_len)
            return left + '|' + right
        else:
            return '(' + gene(len - 2) + ')'
    ans = 'error'
    while 'error' in ans:
        ans = gene(len_)
    return ans

def and_only(len_):
    if len_ % 2 == 0:
        len_ -= 1
    def gene(len):
        if len == 1:
            return np.random.choice(['0', '1'])
        if len == 2:
            return 'error'
        if np.random.rand() < 0.7:
            pos = np.random.randint(1, len - 1)
            left_len = pos
            right_len = len - pos - 1
            left = gene(left_len)
            right = gene(right_len)
            return left + '&' + right
        else:
            return '(' + gene(len - 2) + ')'
    ans = 'error'
    while 'error' in ans:
        ans = gene(len_)
    return ans

def not_only(len_):
    def gene(len):
        if len == 1:
            return np.random.choice(['0', '1'])
        if len == 2:
            return '~' + np.random.choice(['0', '1'])
        if np.random.rand() < 0.5:
            return '~' + gene(len - 1)
        else:
            return '(' + gene(len - 2) + ')'
    return gene(len_)

def small(len_):
    next_char_list = {
        '1': ['|', '&', ')'],
        '0': ['|', '&', ')'],
        '|': ['1', '0', '(', '~'],
        '~': ['1', '0', '(', '~'],
        '&': ['1', '0', '(', '~'],
        '(': ['1', '0', '(', '~'],
        ')': ['|', '&', ')']
    }
    seq = ''
    cnt = 0
    while len(seq) < len_ or seq[-1] in ['(', '|', '&', '~'] or cnt:
        seq = np.random.choice(['1', '0', '(', '~'])
        cnt = seq.count('(')
        while len(seq) < len_ and cnt >= 0:
            seq += np.random.choice(next_char_list[seq[-1]])
            if seq[-1] == '(':
                cnt += 1
            if seq[-1] == ')':
                cnt -= 1
    return seq

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
        return int(calc(seq[:first_or]) or calc(seq[first_or + 1:]))
    if first_and != -1:
        return int(calc(seq[:first_and]) and calc(seq[first_and + 1:]))
    if seq[0] == '~':
        return 1 - calc(seq[1:])
    return calc(seq[1:-1])

or_only_data = []
and_only_data = []
not_only_data = []
small_data = []
for _ in range(count):
    data = or_only(np.random.randint(min_len, max_len))
    or_only_data.append([data, int('1' in data)])
    data = and_only(np.random.randint(min_len, max_len))
    and_only_data.append([data, int(not ('0' in data))])
    data = not_only(np.random.randint(min_len, max_len))
    not_only_data.append([data, int(data.count('~') % 2 == 0 if '1' in data else data.count('~') % 2 == 1)])
    data = small(int(np.log2(np.random.rand() * (pow(1.5, 10) - 1) + 1) / np.log2(1.5)) + 1)
    small_data.append([data, calc(data)])

or_only_df = pd.DataFrame(or_only_data, columns=['seq', 'ans'])
and_only_df = pd.DataFrame(and_only_data, columns=['seq', 'ans'])
not_only_df = pd.DataFrame(not_only_data, columns=['seq', 'ans'])
small_df = pd.DataFrame(small_data, columns=['seq', 'ans'])
or_only_df.to_csv(f'2025-12月/数据生成-特殊数据/仅或运算,maxlen={max_len},size=10^{int(np.log10(count))}.csv', index=False)
and_only_df.to_csv(f'2025-12月/数据生成-特殊数据/仅与运算,maxlen={max_len},size=10^{int(np.log10(count))}.csv', index=False)
not_only_df.to_csv(f'2025-12月/数据生成-特殊数据/仅非运算,maxlen={max_len},size=10^{int(np.log10(count))}.csv', index=False)
small_df.to_csv(f'2025-12月/数据生成-特殊数据/小规模随机表达式,maxlen={min_len},size=10^{int(np.log10(count))}.csv', index=False)