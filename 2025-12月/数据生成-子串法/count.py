import pandas as pd

data = pd.read_csv('2025-12月/数据生成-子串法/子串法,maxlen=15,m=5,size=10^5,test.csv')
cnt_0 = 0
cnt_1 = 0
for i in range(len(data)):
    if data['ans'].values[i] == 1:
        cnt_1 += 1
    else:
        cnt_0 += 1
print(f'0的数量: {cnt_0}, 1的数量: {cnt_1}')