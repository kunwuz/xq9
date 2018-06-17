import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import scale




# 方差
def ptp(column):
    return max(column) - min(column)


# 方差除以评率
def var_divide_count(column):
    return np.var(column) / len(column)


# last one
def last_one(column):
    if len(column) < 1:
        return column
    return column.iloc[-1]


def q25(column):
    return column.quantile(.25)


def q75(column):
    return column.quantile(.75)


def kurt(column):
    return column.kurt()

def add_author_video_feature(df,ids,begin_day,end_day):
    author_video = df[['user_id', 'day','author_id','video_id']]
    # 选时间区间
    author_video=author_video[(author_video.day>=begin_day) & (author_video.day<=end_day)]

    func = ['min', 'max', 'mean', 'median', 'skew', 'std', 'mad', 'count', ptp, np.var,
            var_divide_count, last_one, q25, q75, kurt]
    author = author_video.groupby(['user_id'])['author_id'].agg(func).reset_index()
    author = author.rename(
        columns={'min': 'author_id_min', 'max': 'author_id_max', 'mean': 'author_id_mean', 'median': 'author_id_median',
                 'skew': 'author_id_skew', 'std': 'author_id_std', 'mad': 'author_id_mad', 'count': 'author_id_count',
                 'ptp': 'author_id_ptp', 'var': 'author_id_var', 'var_divide_count': 'author_id_var_divide_count',
                 'last_one': 'author_id_last_one', 'q25': 'author_id_q25', 'q75': 'author_id_q75',
                 'kurt': 'author_id_kurt'})
    video=author_video.groupby(['user_id'])['video_id'].agg(func).reset_index()
    video = video.rename(
        columns={'min': 'video_id_min', 'max': 'video_id_max', 'mean': 'video_id_mean', 'median': 'video_id_median',
                 'skew': 'video_id_skew', 'std': 'video_id_std', 'mad': 'video_id_mad', 'count': 'video_id_count',
                 'ptp': 'video_id_ptp', 'var': 'video_id_var', 'var_divide_count': 'video_id_var_divide_count',
                 'last_one': 'video_id_last_one', 'q25': 'video_id_q25', 'q75': 'video_id_q75',
                 'kurt': 'video_id_kurt'})

    ids=pd.merge(ids, author, how='left', on=['user_id'])
    ids = pd.merge(ids, video, how='left', on=['user_id'])
    return ids



path = os.path.join(os.path.curdir, 'data')
# ------------------train1----------------------
# feature3 train1
user_act_train1 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train1.csv'), encoding='utf-8')
train1 = pd.read_csv(os.path.join(path, 'train_data/data1.csv'), encoding='utf-8')
feature2=pd.read_csv(os.path.join(path, 'train_data/data12.csv'), encoding='utf-8')
ids = pd.DataFrame(train1['user_id'])  # 提取id
begin_day = 7
end_day = 16
feature3=add_author_video_feature(user_act_train1,ids,begin_day=begin_day,end_day=end_day)
feature3=pd.merge(feature2,feature3,how='left',on=['user_id'])
feature3.to_csv(os.path.join(path, 'train_data/data123.csv'), encoding='utf-8', index=False)
print(feature3)

# ------------------train2----------------------
# feature3 train2
user_act_train2 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train2.csv'), encoding='utf-8')
train2 = pd.read_csv(os.path.join(path, 'train_data/data2.csv'), encoding='utf-8')
feature2=pd.read_csv(os.path.join(path, 'train_data/data22.csv'), encoding='utf-8')
ids = pd.DataFrame(train2['user_id'])  # 提取id
begin_day = 14
end_day = 23
feature3=add_author_video_feature(user_act_train2,ids,begin_day=begin_day,end_day=end_day)
feature3=pd.merge(feature2,feature3,how='left',on=['user_id'])
feature3.to_csv(os.path.join(path, 'train_data/data223.csv'), encoding='utf-8', index=False)
print(feature3)

# ------------------train3----------------------
# feature3 train3
user_act_train3 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train3.csv'), encoding='utf-8')
train3 = pd.read_csv(os.path.join(path, 'train_data/data3.csv'), encoding='utf-8')
feature2=pd.read_csv(os.path.join(path, 'train_data/data32.csv'), encoding='utf-8')
ids = pd.DataFrame(train3['user_id'])  # 提取id
begin_day = 15
end_day = 24
feature3=add_author_video_feature(user_act_train3,ids,begin_day=begin_day,end_day=end_day)
feature3=pd.merge(feature2,feature3,how='left',on=['user_id'])
feature3.to_csv(os.path.join(path, 'train_data/data323.csv'), encoding='utf-8', index=False)
print(feature3)

# ------------------train4----------------------
# feature3 train4
user_act_train4 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train4.csv'), encoding='utf-8')
train4 = pd.read_csv(os.path.join(path, 'train_data/data4.csv'), encoding='utf-8')
feature2=pd.read_csv(os.path.join(path, 'train_data/data42.csv'), encoding='utf-8')
ids = pd.DataFrame(train4['user_id'])  # 提取id
begin_day = 20
end_day = 30
feature3=add_author_video_feature(user_act_train4,ids,begin_day=begin_day,end_day=end_day)
feature3=pd.merge(feature2,feature3,how='left',on=['user_id'])
feature3.to_csv(os.path.join(path, 'train_data/data423.csv'), encoding='utf-8', index=False)
print(feature3)




