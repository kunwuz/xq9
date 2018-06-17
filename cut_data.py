import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import scale
import os

path = os.path.join(os.path.curdir, 'data')

user_reg = pd.read_table(
    os.path.join(path, 'user_register_log.txt'),
    names=['user_id', 'day', 'register_type', 'device_type'],
    encoding='utf-8', sep='\t'
)
app = pd.read_table(
    os.path.join(path, 'app_launch_log.txt'),
    names=['user_id', 'day'],
    encoding='utf-8', sep='\t'
)
video = pd.read_table(
    os.path.join(path, 'video_create_log.txt'),
    names=['user_id', 'day'],
    encoding='utf-8', sep='\t'
)
user_act = pd.read_table(
    os.path.join(path, 'user_activity_log.txt'),
    names=['user_id', 'day', 'page', 'videoid', 'authorid', 'action_type'],
    encoding='utf-8', sep='\t'
)
# 归一化
author_video = pd.DataFrame({'author_id': scale(user_act['authorid']), 'video_id': scale(user_act['videoid'])})
user_act = pd.DataFrame(pd.concat([user_act, author_video], axis=1))


# 以时间划分训练集测试集
def cut_data_as_time(begin_day, end_day):
    _user_reg = user_reg[(user_reg['day'] >= begin_day) & (user_reg['day'] <= end_day)]
    _app = app[(app['day'] >= begin_day) & (app['day'] <= end_day)]
    _video = video[(video['day'] >= begin_day) & (video['day'] <= end_day)]
    _user_act = user_act[(user_act['day'] >= begin_day) & (user_act['day'] <= end_day)]
    return _user_reg, _app, _video, _user_act


# 1
begin_day = 1
end_day = 16
user_reg_train1, app_train1, video_train1, user_act_train1 = cut_data_as_time(begin_day, end_day)
# print(user_reg_train1)
begin_day = 17
end_day = 23
user_reg_test1, app_test1, video_test1, user_act_test1 = cut_data_as_time(begin_day, end_day)

# 8-23 预测 24-30
begin_day = 1
end_day = 18
user_reg_train2, app_train2, video_train2, user_act_train2 = cut_data_as_time(begin_day, end_day)
begin_day = 19
end_day = 25
user_reg_test2, app_test2, video_test2, user_act_test2 = cut_data_as_time(begin_day, end_day)

# 10-20 预测22-30
begin_day = 1
end_day = 20
user_reg_train3, app_train3, video_train3, user_act_train3 = cut_data_as_time(begin_day, end_day)
begin_day = 21
end_day = 27
user_reg_test3, app_test3, video_test3, user_act_test3 = cut_data_as_time(begin_day, end_day)

# 5
begin_day = 1
end_day = 22
user_reg_train5, app_train5, video_train5, user_act_train5 = cut_data_as_time(begin_day, end_day)
begin_day = 23
end_day = 29
user_reg_test5, app_test5, video_test5, user_act_test5 = cut_data_as_time(begin_day, end_day)

# 6
begin_day = 1
end_day = 24
user_reg_train6, app_train6, video_train6, user_act_train6 = cut_data_as_time(begin_day, end_day)
begin_day = 25
end_day = 30
user_reg_test6, app_test6, video_test6, user_act_test6 = cut_data_as_time(begin_day, end_day)

# 7
begin_day = 10
end_day = 23
user_reg_train7, app_train7, video_train7, user_act_train7 = cut_data_as_time(begin_day, end_day)
begin_day = 24
end_day = 30
user_reg_test7, app_test7, video_test7, user_act_test7 = cut_data_as_time(begin_day, end_day)

# 1-30 作为提交
begin_day = 1
end_day = 30
user_reg_train4, app_train4, video_train4, user_act_train4 = cut_data_as_time(begin_day, end_day)


# 打标签
def get_label(user_reg_train, app_train, video_train, user_act_train, user_reg_test, app_test, video_test,
              user_act_test):
    train_id = DataFrame(pd.concat([user_reg_train, app_train, video_train, user_act_train]))
    train_id = list(train_id.drop_duplicates(subset=['user_id'])['user_id'])
    test_id = DataFrame(pd.concat([user_reg_test, app_test, video_test, user_act_test]))
    test_id = list(test_id.drop_duplicates(subset=['user_id'])['user_id'])
    print(len(train_id))
    print(len(test_id))

    # 前后都有打1
    label = []
    for i in train_id:
        if i in test_id:
            label.append(1)
        else:
            label.append(0)
    print(len(label))
    label_train = DataFrame({'user_id': train_id, 'label': label})
    return label_train


def data_to_csv(i, user_reg_train, app_train, video_train, user_act_train, user_reg_test, app_test, video_test,
                user_act_test):
    # 创建准备数据文件夹
    if not os.path.exists(os.path.join(path, 'cut_data')):
        os.makedirs(os.path.join(path, 'cut_data'))

    # -------------train-------------
    user_reg_train.to_csv(os.path.join(path, 'cut_data/user_reg_train%s.csv' % i), encoding='utf-8', index=False)
    app_train.to_csv(os.path.join(path, 'cut_data/app_train%s.csv' % i), encoding='utf-8', index=False)
    video_train.to_csv(os.path.join(path, 'cut_data/video_train%s.csv' % i), encoding='utf-8', index=False)
    user_act_train.to_csv(os.path.join(path, 'cut_data/user_act_train%s.csv' % i), encoding='utf-8', index=False)
    label_train = get_label(
        user_reg_train, app_train, video_train, user_act_train, user_reg_test, app_test, video_test,
        user_act_test)
    label_train.to_csv(os.path.join(path, 'cut_data/label_train%s.csv' % i), encoding='utf-8', index=False)
    del user_reg_train
    del app_train
    del video_train
    del user_act_train


# 创建准备数据文件夹
if not os.path.exists(os.path.join(path, 'cut_data')):
    os.makedirs(os.path.join(path, 'cut_data'))

# -------------train1-------------
# user_reg_train1.to_csv(os.path.join(path, 'cut_data/user_reg_train1.csv'), encoding='utf-8', index=False)
# app_train1.to_csv(os.path.join(path, 'cut_data/app_train1.csv'), encoding='utf-8', index=False)
# video_train1.to_csv(os.path.join(path, 'cut_data/video_train1.csv'), encoding='utf-8', index=False)
# user_act_train1.to_csv(os.path.join(path, 'cut_data/user_act_train1.csv'), encoding='utf-8', index=False)
# label_train1 = get_label(
#     user_reg_train1, app_train1, video_train1, user_act_train1, user_reg_test1, app_test1, video_test1, user_act_test1)
# label_train1.to_csv(os.path.join(path, 'cut_data/label_train1.csv'), encoding='utf-8', index=False)
data_to_csv(1, user_reg_train1, app_train1, video_train1, user_act_train1, user_reg_test1, app_test1, video_test1,
            user_act_test1)

# ----------train2----------------
data_to_csv(2, user_reg_train2, app_train2, video_train2, user_act_train2, user_reg_test2, app_test2, video_test2,
            user_act_test2)

# ----------train3----------------
data_to_csv(3, user_reg_train3, app_train3, video_train3, user_act_train3, user_reg_test3, app_test3, video_test3,
            user_act_test3)

# ----------train5----------------
data_to_csv(5, user_reg_train5, app_train5, video_train5, user_act_train5, user_reg_test5, app_test5, video_test5,
            user_act_test5)

# ----------train6----------------
data_to_csv(6, user_reg_train6, app_train6, video_train6, user_act_train6, user_reg_test6, app_test6, video_test6,
            user_act_test6)

# ----------train7----------------
data_to_csv(7, user_reg_train7, app_train7, video_train7, user_act_train7, user_reg_test7, app_test7, video_test7,
            user_act_test7)

# -----------submit data-------------
user_reg_train4.to_csv(os.path.join(path, 'cut_data/user_reg_train4.csv'), encoding='utf-8', index=False)
app_train4.to_csv(os.path.join(path, 'cut_data/app_train4.csv'), encoding='utf-8', index=False)
video_train4.to_csv(os.path.join(path, 'cut_data/video_train4.csv'), encoding='utf-8', index=False)
user_act_train4.to_csv(os.path.join(path, 'cut_data/user_act_train4.csv'), encoding='utf-8', index=False)
