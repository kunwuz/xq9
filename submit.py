import os
from pandas import DataFrame
import pandas as pd
path=os.path.join(os.path.curdir,'data')


train12 = pd.read_csv(os.path.join(path, 'train_data/data12.csv'), encoding='utf-8')
train22 = pd.read_csv(os.path.join(path, 'train_data/data22.csv'), encoding='utf-8')
train32=pd.read_csv(os.path.join(path, 'train_data/data32.csv'), encoding='utf-8')
print(train12.shape)

train12_22=DataFrame(pd.concat([train12,train22],axis=0))
train12_22=train12_22.drop_duplicates()
train12_22.to_csv(os.path.join(path,'train_data/data12_22.csv'),encoding='utf-8',index=None)
print(train12_22.shape)

train12_32=DataFrame(pd.concat([train12,train32],axis=0))
train12_32=train12_32.drop_duplicates()
train12_32.to_csv(os.path.join(path,'train_data/data12_32.csv'),encoding='utf-8',index=None)

train22_32=DataFrame(pd.concat([train22,train32],axis=0))
train22_32=train22_32.drop_duplicates()
train22_32.to_csv(os.path.join(path,'train_data/data22_32.csv'),encoding='utf-8',index=None)

train12_22_32=DataFrame(pd.concat([train12,train22,train32],axis=0))
train12_22_32=train12_22_32.drop_duplicates()
train12_22_32.to_csv(os.path.join(path,'train_data/data12_22_32.csv'),encoding='utf-8',index=None)
print(train12_22_32.shape)

# ------------------------
train123 = pd.read_csv(os.path.join(path, 'train_data/data123.csv'), encoding='utf-8')
train223 = pd.read_csv(os.path.join(path, 'train_data/data223.csv'), encoding='utf-8')
train323=pd.read_csv(os.path.join(path, 'train_data/data323.csv'), encoding='utf-8')
print(train123.shape)

train123_223=DataFrame(pd.concat([train123,train223],axis=0))
train123_223=train123_223.drop_duplicates()
train123_223.to_csv(os.path.join(path,'train_data/data123_223.csv'),encoding='utf-8',index=None)
print(train123_223.shape)

train123_323=DataFrame(pd.concat([train123,train323],axis=0))
train123_323=train123_323.drop_duplicates()
train123_323.to_csv(os.path.join(path,'train_data/data123_323.csv'),encoding='utf-8',index=None)


train223_323=DataFrame(pd.concat([train223,train323],axis=0))
train223_323=train223_323.drop_duplicates()
train223_323.to_csv(os.path.join(path,'train_data/data223_323.csv'),encoding='utf-8',index=None)

train123_223_323=DataFrame(pd.concat([train123,train223,train323],axis=0))
train123_223_323=train123_223_323.drop_duplicates()
train123_223_323.to_csv(os.path.join(path,'train_data/data123_223_323.csv'),encoding='utf-8',index=None)
print(train123_223_323.shape)

