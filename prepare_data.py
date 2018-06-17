# coding:utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import matplotlib.pyplot as plt

path=os.path.join(os.path.curdir,'data')



user_reg=pd.read_table(
    os.path.join(path,'user_register_log.txt'),
    names=['user_id','day','register_type','device_type'],
    encoding='utf-8',sep='\t'
)
# app=pd.read_table(
#     os.path.join(path,'app_launch_log.txt'),
#     names=['user_id','day'],
#     encoding='utf-8',sep='\t'
# )
# video=pd.read_table(
#     os.path.join(path,'video_create_log.txt'),
#     names=['user_id','day'],
#     encoding='utf-8',sep='\t'
# )
# user_act=pd.read_table(
#     os.path.join(path,'user_activity_log.txt'),
#     names=['user_id','day','page','video_id','author_id','action_type'],
#     encoding='utf-8',sep='\t'
# )

#(51709,4)
print(user_reg.shape)
#(251943,2)
# print(app.shape)
# # (35151,2)
# print(video.shape)
# # (20051557,6)
# print(user_act.shape)


# 下面说明所有id都和注册一样
# ids=DataFrame(pd.concat([app.user_id,video.user_id,user_act.user_id],axis=0))
# ids=ids.drop_duplicates()
# ids=ids[~ids['user_id'].isin(user_reg.user_id)]
# print(ids.shape)
# print(ids.head())

# 不同类型数
# print(len(user_reg['device_type'].unique()))
# app_day=app.groupby(['user_id','day']).size().reset_index()
# print(app_day)

count=user_reg.groupby('day')['user_id'].agg(['count']).reset_index()
print(count)
plt.plot(count)
plt.show()




# userPre = user_act[user_act.day>=24]
# sub = userPre[['user_id']].drop_duplicates()
# sub=sub.sort_values('user_id')
# sub.to_csv(os.path.join(path,'submission1.csv'),encoding='utf-8',index=None,header=None)
#
#
# s1=pd.read_csv(os.path.join(path,'submission1.csv'),encoding='utf-8')
# s3=pd.read_csv(os.path.join(path,'submission3.csv'),encoding='utf-8')
#
# s4=DataFrame(pd.concat([s1,s3],axis=0)).drop_duplicates()
# s4.to_csv(os.path.join(path,'submission4.csv'),encoding='utf-8',index=None,header=None)