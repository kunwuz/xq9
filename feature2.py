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


def past_days(df, ids, begin_day, end_day):
    for i in range(begin_day, end_day + 1):
        d = df[df['day'] == i].groupby(['user_id', 'day']).count().reset_index()
        d = d[['user_id', 'page']].rename(columns={'page': 'pastday%s' % (end_day - i)})
        # print(d)
        ids = pd.merge(ids, d, how='left', on=['user_id'])
    return ids


def past_days_day(df, ids, end_day):
    begin_day = end_day - 6
    for i in range(begin_day, end_day + 1):
        d = df[df['day'] == i].groupby(['user_id', 'day']).count().reset_index()
        d = d[['user_id', 'page']].rename(columns={'page': 'pastday%s' % (end_day - i)})
        # print(d)
        ids = pd.merge(ids, d, how='left', on=['user_id'])
    return ids


def past_days_action(df, ids, end_day):
    begin_day = end_day - 6
    for day in range(begin_day, end_day + 1):
        for type in range(6):
            d = pd.DataFrame(df[(df['day'] == day) & (df['action_type'] == type)][['user_id', 'dtype_count']])
            d = d.rename(columns={'dtype_count': 'past%s_atype%s' % (end_day - day, type)})
            ids = pd.merge(ids, d, how='left', on=['user_id'])
    return ids


def past_days_page(df, ids, end_day):
    begin_day = end_day - 6
    for day in range(begin_day, end_day + 1):
        for type in range(5):
            d = pd.DataFrame(df[(df['day'] == day) & (df['page'] == type)][['user_id', 'page_count']])
            d = d.rename(columns={'page_count': 'past%s_ptype%s' % (end_day - day, type)})
            ids = pd.merge(ids, d, how='left', on=['user_id'])
    return ids


# # 下面效率不高
# def each_page_count1(df, ids, begin_day, end_day):
#     for type in range(5):
#         d = df[df.page == type].groupby(['user_id', 'day'])['page'].agg(['count']).reset_index()
#         d = pd.DataFrame(d[(d['day'] >= begin_day) & (d['day'] <= end_day)][['user_id', 'count']])
#         d = d.groupby('user_id')['count'].agg(['sum']).reset_index()
#         d = d.rename(columns={'sum': '%s_page_count' % type})
#         ids = pd.merge(ids, d, how='left', on=['user_id'])
#     return ids
#
#
# def each_act_count1(df, ids, begin_day, end_day):
#     for type in range(6):
#         d = df[df.action_type == type].groupby(['user_id', 'day'])['action_type'].agg(['count']).reset_index()
#         d = pd.DataFrame(d[(d['day'] >= begin_day) & (d['day'] <= end_day)][['user_id', 'count']])
#         d = d.groupby('user_id')['count'].agg(['sum']).reset_index()
#         d = d.rename(columns={'sum': '%s_act_count' % type})
#         ids = pd.merge(ids, d, how='left', on=['user_id'])
#     return ids

def each_page_count(df, ids, begin_day, end_day):
    df = df[(df.day >= begin_day) & (df.day <= end_day)]
    for type in range(5):
        d = df[df.page == type].groupby(['user_id'])['page'].agg(['count']).reset_index()
        d = d.rename(columns={'count': '%s_page_count' % type})
        ids = pd.merge(ids, d, how='left', on=['user_id'])
    return ids


def each_act_count(df, ids, begin_day, end_day):
    df = df[(df.day >= begin_day) & (df.day <= end_day)]
    for type in range(6):
        d = df[df.page == type].groupby(['user_id'])['action_type'].agg(['count']).reset_index()
        d = d.rename(columns={'count': '%s_act_count' % type})
        ids = pd.merge(ids, d, how='left', on=['user_id'])
    return ids


def page_act_stat(df, begin_day, end_day):
    df = df[(df.day >= begin_day) & (df.day <= end_day)]
    func = ['min', 'max', 'mean', 'median', 'skew', 'std', 'mad', 'count', ptp, np.var,
            var_divide_count, last_one, q25, q75, kurt]
    page_stat = df.groupby(['user_id'])['page'].agg(func).reset_index()  ## page 统计
    page_stat = page_stat.rename(
        columns={'min': 'page_stat_min', 'max': 'page_stat_max', 'mean': 'page_stat_mean', 'median': 'page_stat_median',
                 'skew': 'page_stat_skew', 'std': 'page_stat_std', 'mad': 'page_stat_mad', 'count': 'page_stat_count',
                 'ptp': 'page_stat_ptp', 'var': 'page_stat_var', 'var_divide_count': 'page_stat_var_divide_count',
                 'last_one': 'page_stat_last_one', 'q25': 'page_stat_q25', 'q75': 'page_stat_q75',
                 'kurt': 'page_stat_kurt'})
    act_stat = df.groupby(['user_id'])['action_type'].agg(func).reset_index()  ## act 统计
    act_stat = act_stat.rename(
        columns={'min': 'act_stat_min', 'max': 'act_stat_max', 'mean': 'act_stat_mean', 'median': 'act_stat_median',
                 'skew': 'act_stat_skew', 'std': 'act_stat_std', 'mad': 'act_stat_mad', 'count': 'act_stat_count',
                 'ptp': 'act_stat_ptp', 'var': 'act_stat_var', 'var_divide_count': 'act_stat_var_divide_count',
                 'last_one': 'act_stat_last_one', 'q25': 'act_stat_q25', 'q75': 'act_stat_q75',
                 'kurt': 'act_stat_kurt'})
    return page_stat, act_stat


def merge_feature(i, begin_day, end_day):
    user_act_train = pd.read_csv(os.path.join(path, 'cut_data/user_act_train%s.csv' % i), encoding='utf-8')
    train = pd.read_csv(os.path.join(path, 'train_data/data%s.csv' % i), encoding='utf-8')

    ids = pd.DataFrame(train['user_id'])  # 提取id
    day_freq = user_act_train[['user_id', 'day', 'page']]  # day切片

    page_act = user_act_train[['user_id', 'day', 'page', 'action_type']]  # 切片
    page = page_act.groupby(['user_id', 'day', 'page']).count().reset_index().rename(
        columns={'action_type': 'page_count'})  # page 每天page
    act_type_count = page_act.groupby(['user_id', 'day', 'action_type']).count().reset_index().rename(
        columns={'page': 'dtype_count'})  # action

    day_data = past_days_day(day_freq, ids, end_day)  ## 过去几天每天活动次数
    page_data = past_days_page(page, ids, end_day)  ## 过去几天每天page count
    act_data = past_days_action(act_type_count, ids, end_day)  ## 过去几天每天action_type count

    page_count = each_page_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种page count
    act_count = each_act_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种action_type count

    page_stat, act_stat = page_act_stat(page_act, begin_day=begin_day, end_day=end_day)  ## page he act tongji

    feature2 = pd.merge(train1, day_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_stat, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_stat, how='left', on=['user_id'])

    return feature2


if __name__ == '__main__':
    path = os.path.join(os.path.curdir, 'data')
    # ------------------train1----------------------
    # feature2 train1
    user_act_train1 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train1.csv'), encoding='utf-8')
    train1 = pd.read_csv(os.path.join(path, 'train_data/data1.csv'), encoding='utf-8')

    ids = pd.DataFrame(train1['user_id'])  # 提取id
    day_freq = user_act_train1[['user_id', 'day', 'page']]  # day切片

    page_act = user_act_train1[['user_id', 'day', 'page', 'action_type']]  # 切片
    page = page_act.groupby(['user_id', 'day', 'page']).count().reset_index().rename(
        columns={'action_type': 'page_count'})  # page 每天page
    act_type_count = page_act.groupby(['user_id', 'day', 'action_type']).count().reset_index().rename(
        columns={'page': 'dtype_count'})  # action
    begin_day = 7
    end_day = 16
    day_data = past_days_day(day_freq, ids, end_day)  ## 过去几天每天活动次数
    page_data = past_days_page(page, ids, end_day)  ## 过去几天每天page count
    act_data = past_days_action(act_type_count, ids, end_day)  ## 过去几天每天action_type count

    page_count = each_page_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种page count
    act_count = each_act_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种action_type count

    page_stat, act_stat = page_act_stat(page_act, begin_day=begin_day, end_day=end_day)  ## page he act tongji

    feature2 = pd.merge(train1, day_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_stat, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_stat, how='left', on=['user_id'])
    print(feature2)
    feature2.to_csv(os.path.join(path, 'train_data/data12.csv'), encoding='utf-8', index=False)
    #
    # ------------------train2----------------------
    # feature2 train2
    user_act_train2 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train2.csv'), encoding='utf-8')
    train2 = pd.read_csv(os.path.join(path, 'train_data/data2.csv'), encoding='utf-8')

    ids = pd.DataFrame(train2['user_id'])  # 提取id
    day_freq = user_act_train2[['user_id', 'day', 'page']]  # day切片

    page_act = user_act_train2[['user_id', 'day', 'page', 'action_type']]  # 切片
    page = page_act.groupby(['user_id', 'day', 'page']).count().reset_index().rename(
        columns={'action_type': 'page_count'})  # page 每天page
    act_type_count = page_act.groupby(['user_id', 'day', 'action_type']).count().reset_index().rename(
        columns={'page': 'dtype_count'})  # action

    begin_day = 14
    end_day = 23
    day_data = past_days_day(day_freq, ids, end_day)  ## 过去几天每天活动次数
    page_data = past_days_page(page, ids, end_day)  ## 过去几天每天page count
    act_data = past_days_action(act_type_count, ids, end_day)  ## 过去几天每天action_type count
    page_count = each_page_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种page count
    act_count = each_act_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种action_type count

    page_stat, act_stat = page_act_stat(page_act, begin_day=begin_day, end_day=end_day)  ## page he act tongji

    feature2 = pd.merge(train2, day_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_stat, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_stat, how='left', on=['user_id'])
    print(feature2)
    feature2.to_csv(os.path.join(path, 'train_data/data22.csv'), encoding='utf-8', index=False)

    # ------------------train3----------------------
    # feature2 train3
    user_act_train3 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train3.csv'), encoding='utf-8')
    train3 = pd.read_csv(os.path.join(path, 'train_data/data3.csv'), encoding='utf-8')

    ids = pd.DataFrame(train3['user_id'])  # 提取id
    day_freq = user_act_train3[['user_id', 'day', 'page']]  # day切片

    page_act = user_act_train3[['user_id', 'day', 'page', 'action_type']]  # 切片
    page = page_act.groupby(['user_id', 'day', 'page']).count().reset_index().rename(
        columns={'action_type': 'page_count'})  # page 每天page
    act_type_count = page_act.groupby(['user_id', 'day', 'action_type']).count().reset_index().rename(
        columns={'page': 'dtype_count'})  # action

    begin_day = 15
    end_day = 24
    day_data = past_days_day(day_freq, ids, end_day)  ## 过去几天每天活动次数
    page_data = past_days_page(page, ids, end_day)  ## 过去几天每天page count
    act_data = past_days_action(act_type_count, ids, end_day)  ## 过去几天每天action_type count

    page_count = each_page_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种page count
    act_count = each_act_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种action_type count
    page_stat, act_stat = page_act_stat(page_act, begin_day=begin_day, end_day=end_day)  ## page he act tongji

    feature2 = pd.merge(train3, day_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_stat, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_stat, how='left', on=['user_id'])
    print(feature2)
    feature2.to_csv(os.path.join(path, 'train_data/data32.csv'), encoding='utf-8', index=False)

    # ------------------submit----------------------
    # feature2 submit
    user_act_train4 = pd.read_csv(os.path.join(path, 'cut_data/user_act_train4.csv'), encoding='utf-8')
    train4 = pd.read_csv(os.path.join(path, 'train_data/data4.csv'), encoding='utf-8')

    ids = pd.DataFrame(train4['user_id'])  # 提取id
    day_freq = user_act_train4[['user_id', 'day', 'page']]  # day切片

    page_act = user_act_train4[['user_id', 'day', 'page', 'action_type']]  # 切片
    page = page_act.groupby(['user_id', 'day', 'page']).count().reset_index().rename(
        columns={'action_type': 'page_count'})  # page 每天page
    act_type_count = page_act.groupby(['user_id', 'day', 'action_type']).count().reset_index().rename(
        columns={'page': 'dtype_count'})  # action

    begin_day = 20
    end_day = 30
    day_data = past_days_day(day_freq, ids, end_day)  ## 过去几天每天活动次数
    page_data = past_days_page(page, ids, end_day)  ## 过去几天每天page count
    act_data = past_days_action(act_type_count, ids, end_day)  ## 过去几天每天action_type count

    page_count = each_page_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种page count
    act_count = each_act_count(page_act, ids, begin_day=begin_day, end_day=end_day)  ## 过去几天每天每种action_type count
    page_stat, act_stat = page_act_stat(page_act, begin_day=begin_day, end_day=end_day)  ## page he act tongji

    feature2 = pd.merge(train4, day_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_data, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_count, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, page_stat, how='left', on=['user_id'])
    feature2 = pd.merge(feature2, act_stat, how='left', on=['user_id'])
    print(feature2)
    feature2.to_csv(os.path.join(path, 'train_data/data42.csv'), encoding='utf-8', index=False)

    # ------------------train5----------------------
    # feature2 train5
    begin_day = 15
    end_day = 24
    feature2 = merge_feature(5,begin_day=begin_day,end_day=end_day)
    print(feature2)
    feature2.to_csv(os.path.join(path, 'train_data/data52.csv'), encoding='utf-8', index=False)

