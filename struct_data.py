'''
将数据排序后转csv，但是发现读取速度变慢
'''
import pandas as pd
import os

path = os.path.join(os.path.curdir, 'data/datacsv')

if not os.path.exists(os.path.join(path, 'user_reg.csv')):
    user_reg = pd.read_table(
        os.path.join(path, 'user_register_log.txt'),
        names=['user_id', 'register_day', 'register_type', 'device_type'],
        encoding='utf-8', sep='\t'
    )
    user_reg = user_reg.sort_values('user_id')
    user_reg.to_csv(os.path.join(path, 'user_reg.csv'), encoding='utf-8', index=False)

if not os.path.exists(os.path.join(path, 'app.csv')):
    app = pd.read_table(
        os.path.join(path, 'app_launch_log.txt'),
        names=['user_id', 'day'],
        encoding='utf-8', sep='\t'
    )
    app = app.sort_values('user_id')
    app.to_csv(os.path.join(path, 'app.csv'), encoding='utf-8', index=False)

if not os.path.exists(os.path.join(path, 'video.csv')):
    video = pd.read_table(
        os.path.join(path, 'video_create_log.txt'),
        names=['user_id', 'day'],
        encoding='utf-8', sep='\t'
    )
    video = video.sort_values('user_id')
    video.to_csv(os.path.join(path, 'video.csv'), encoding='utf-8', index=False)

if not os.path.exists(os.path.join(path, 'user_act.csv')):
    user_act = pd.read_table(
        os.path.join(path, 'user_activity_log.txt'),
        names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'],
        encoding='utf-8', sep='\t'
    )
    user_act = user_act.sort_values('user_id')
    user_act.to_csv(os.path.join(path, 'user_act.csv'), encoding='utf-8', index=False)
