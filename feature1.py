# coding:utf-8

import numpy as np
import pandas as pd
import os


# 方差
def ptp(column):
    return max(column) - min(column)


# 方差除以评率
def var_divide_count(column):
    return np.var(column) / len(column)


def last_one(column):
    if len(column) < 1:
        return column
    return column.iloc[-1]
def last_one_minus_reg():
    return

def q25(column):
    return column.quantile(.25)


def q75(column):
    return column.quantile(.75)


def kurt(column):
    return column.kurt()


def get_features(user_reg_train, app_train, video_train, user_act_train, begin_day=10, end_day=16):
    # APP 启动次数，平均次数，方差，启动天数，最大值，最小值，连续几天启动总次数，平均次数，某一天启动次数
    # agg要使用的函数
    func = ['min', 'max', 'mean', 'median', 'skew', 'std', 'mad', 'count', ptp, np.var,
            var_divide_count, last_one, q25, q75, kurt]
    # 提取前一段时间段的特征
    # user_reg_train = user_reg_train[(user_reg_train['day'] >= begin_day) & (user_reg_train['day'] <= end_day)]
    app_train = app_train[(app_train['day'] >= begin_day) & (app_train['day'] <= end_day)]
    video_train = video_train[(video_train['day'] >= begin_day) & (video_train['day'] <= end_day)]
    user_act_train = user_act_train[(user_act_train['day'] >= begin_day) & (user_act_train['day'] <= end_day)]

    reg = user_reg_train.sort_values(by='user_id')
    app = app_train.groupby('user_id')['day'].agg(func).reset_index()
    app = app.rename(
        columns={'min': 'app_min', 'max': 'app_max', 'mean': 'app_mean', 'median': 'app_median',
                 'skew': 'app_skew', 'std': 'app_std', 'mad': 'app_mad', 'count': 'app_count',
                 'ptp': 'app_ptp', 'var': 'app_var', 'var_divide_count': 'app_var_divide_count',
                 'last_one': 'app_last_one', 'q25': 'app_q25', 'q75': 'app_q75', 'kurt': 'app_kurt'})
    df = pd.merge(reg, app, how='left', on=['user_id'])

    video = video_train.groupby('user_id')['day'].agg(func).reset_index()
    video = video.rename(
        columns={'min': 'video_min', 'max': 'video_max', 'mean': 'video_mean', 'median': 'video_median',
                 'skew': 'video_skew', 'std': 'video_std', 'mad': 'video_mad', 'count': 'video_count',
                 'ptp': 'video_ptp', 'var': 'video_var', 'var_divide_count': 'video_var_divide_count',
                 'last_one': 'video_last_one', 'q25': 'video_q25', 'q75': 'video_q75', 'kurt': 'video_kurt'})
    df = pd.merge(df, video, how='left', on=['user_id'])

    act_day = user_act_train[['user_id', 'day']]
    act_day = act_day.groupby('user_id')['day'].agg(func).reset_index()
    act_day = act_day.rename(
        columns={'min': 'act_min', 'max': 'act_max', 'mean': 'act_mean', 'median': 'act_median',
                 'skew': 'act_skew', 'std': 'act_std', 'mad': 'act_mad', 'count': 'act_count',
                 'ptp': 'act_ptp', 'var': 'act_var', 'var_divide_count': 'act_var_divide_count',
                 'last_one': 'act_last_one', 'q25': 'act_q25', 'q75': 'act_q75', 'kurt': 'act_kurt'})
    df = pd.merge(df, act_day, how='left', on=['user_id'])

    df['app_last_minus_reg']=df['app_last_one']-df['day']
    df['video_last_minus_reg'] = df['video_last_one'] - df['day']
    df['act_last_minus_reg'] = df['act_last_one'] - df['day']

    return df


if __name__ == '__main__':
    path = os.path.join(os.path.curdir, 'data')
    # 创建训练数据文件夹
    if not os.path.exists(os.path.join(path, 'train_data')):
        os.makedirs(os.path.join(path, 'train_data'))

    # -----------train1--------------
    # train1 以1-16天，预测17-23天
    user_reg_train1 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train1.csv'), encoding='utf-8')
    app_train1 = pd.read_csv(os.path.join(path, 'cut_data/app_train1.csv'), encoding='utf-8')
    video_train1 = pd.read_csv(os.path.join(path, 'cut_data/video_train1.csv'), encoding='utf-8')
    user_act_train1 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train1.csv'), encoding='utf-8')
    label_train1 = pd.read_csv(os.path.join(path, 'cut_data/label_train1.csv'), encoding='utf-8')
    # 产生trian1
    train1 = get_features(user_reg_train1, app_train1, video_train1, user_act_train1, begin_day=7, end_day=16)
    train1 = pd.merge(train1, label_train1, how='left', on=['user_id'])
    train1.to_csv(os.path.join(path, 'train_data/data1.csv'), encoding='utf-8', index=False)

    # -----------train2--------------
    # train2 8-23 预测 24-30
    user_reg_train2 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train2.csv'), encoding='utf-8')
    app_train2 = pd.read_csv(os.path.join(path, 'cut_data/app_train2.csv'), encoding='utf-8')
    video_train2 = pd.read_csv(os.path.join(path, 'cut_data/video_train2.csv'), encoding='utf-8')
    user_act_train2 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train2.csv'), encoding='utf-8')
    label_train2 = pd.read_csv(os.path.join(path, 'cut_data/label_train2.csv'), encoding='utf-8')
    # 生产train2
    train2 = get_features(user_reg_train2, app_train2, video_train2, user_act_train2, begin_day=14, end_day=23)
    train2 = pd.merge(train2, label_train2, how='left', on=['user_id'])
    train2.to_csv(os.path.join(path, 'train_data/data2.csv'), encoding='utf-8', index=False)

    # ------------train3-------------
    # train3 16-23 预测 24-30
    user_reg_train3 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train3.csv'), encoding='utf-8')
    app_train3 = pd.read_csv(os.path.join(path, 'cut_data/app_train3.csv'), encoding='utf-8')
    video_train3 = pd.read_csv(os.path.join(path, 'cut_data/video_train3.csv'), encoding='utf-8')
    user_act_train3 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train3.csv'), encoding='utf-8')
    label_train3 = pd.read_csv(os.path.join(path, 'cut_data/label_train3.csv'), encoding='utf-8')
    # 生产train3
    train3 = get_features(user_reg_train3, app_train3, video_train3, user_act_train3, begin_day=13, end_day=24)
    train3 = pd.merge(train3, label_train3, how='left', on=['user_id'])
    train3.to_csv(os.path.join(path, 'train_data/data3.csv'), encoding='utf-8', index=False)

    # -----------submit--------------
    # submit
    user_reg_train4 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train4.csv'), encoding='utf-8')
    app_train4 = pd.read_csv(os.path.join(path, 'cut_data/app_train4.csv'), encoding='utf-8')
    video_train4 = pd.read_csv(os.path.join(path, 'cut_data/video_train4.csv'), encoding='utf-8')
    user_act_train4 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train4.csv'), encoding='utf-8')
    # 生产train4 1 -30 作为测试
    train4 = get_features(user_reg_train4, app_train4, video_train4, user_act_train4, begin_day=20, end_day=30)
    train4.to_csv(os.path.join(path, 'train_data/data4.csv'), encoding='utf-8', index=False)

    # ------------train5-------------
    # train5 16-25 预测 24-50
    user_reg_train5 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train5.csv'), encoding='utf-8')
    app_train5 = pd.read_csv(os.path.join(path, 'cut_data/app_train5.csv'), encoding='utf-8')
    video_train5 = pd.read_csv(os.path.join(path, 'cut_data/video_train5.csv'), encoding='utf-8')
    user_act_train5 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train5.csv'), encoding='utf-8')
    label_train5 = pd.read_csv(os.path.join(path, 'cut_data/label_train5.csv'), encoding='utf-8')
    # 生产train5
    train5 = get_features(user_reg_train5, app_train5, video_train5, user_act_train5, begin_day=15, end_day=24)
    train5 = pd.merge(train5, label_train5, how='left', on=['user_id'])
    train5.to_csv(os.path.join(path, 'train_data/data5.csv'), encoding='utf-8', index=False)

 # ------------train6-------------
    # train6 16-26 预测 24-60
    user_reg_train6 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train6.csv'), encoding='utf-8')
    app_train6 = pd.read_csv(os.path.join(path, 'cut_data/app_train6.csv'), encoding='utf-8')
    video_train6 = pd.read_csv(os.path.join(path, 'cut_data/video_train6.csv'), encoding='utf-8')
    user_act_train6 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train6.csv'), encoding='utf-8')
    label_train6 = pd.read_csv(os.path.join(path, 'cut_data/label_train6.csv'), encoding='utf-8')
    # 生产train6
    train6 = get_features(user_reg_train6, app_train6, video_train6, user_act_train6, begin_day=16, end_day=24)
    train6 = pd.merge(train6, label_train6, how='left', on=['user_id'])
    train6.to_csv(os.path.join(path, 'train_data/data6.csv'), encoding='utf-8', index=False)

 # ------------train7-------------
    # train7 16-27 预测 24-70
    user_reg_train7 = pd.read_csv(os.path.join(path, 'cut_data/user_reg_train7.csv'), encoding='utf-8')
    app_train7 = pd.read_csv(os.path.join(path, 'cut_data/app_train7.csv'), encoding='utf-8')
    video_train7 = pd.read_csv(os.path.join(path, 'cut_data/video_train7.csv'), encoding='utf-8')
    user_act_train7 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train7.csv'), encoding='utf-8')
    label_train7 = pd.read_csv(os.path.join(path, 'cut_data/label_train7.csv'), encoding='utf-8')
    # 生产train7
    train7 = get_features(user_reg_train7, app_train7, video_train7, user_act_train7, begin_day=17, end_day=24)
    train7 = pd.merge(train7, label_train7, how='left', on=['user_id'])
    train7.to_csv(os.path.join(path, 'train_data/data7.csv'), encoding='utf-8', index=False)



