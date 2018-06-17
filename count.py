# coding:utf-8

from pandas import DataFrame
import numpy as np

df = DataFrame({'key1': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c'],
                'key2': ['one', 's', 'two', 's', 'one', 's', 'two', 'one', 'one', 's'],
                'data1': [1, 2, 6, 3, 2, 1, 1, 5, None, np.nan],
                'data2': [1, 2, 6, 5, None, np.nan, 3, 2, 1, 1]
                })

print(df)
# a=df['data1']
# print(a.quantile(.50))
# def q25(c):
#     return c.quantile(.50)
# def kurt(c):
#     return c.kurt()
# a = df.groupby(['key1'])['data1'].agg(['size','count',kurt,q25]).reset_index()
# print(a)

df['minus']=df['data1']
print(df)












