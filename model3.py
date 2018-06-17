import lightgbm as lgb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from feature_name import feature

path = os.path.join(os.path.curdir, 'data')

# ------------load data----------
# train
train = pd.read_csv(os.path.join(path, 'train_data/data223_323.csv'), encoding='utf-8')
X = train.drop(['user_id', 'label'], axis=1)
y = train['label']
# valid
valid_data = pd.read_csv(os.path.join(path, 'train_data/data323.csv'), encoding='utf-8')
_valid_data = valid_data.drop(['user_id', 'label'], axis=1)
# submit
test_data = pd.read_csv(os.path.join(path, 'train_data/data423.csv'), encoding='utf-8')
test = test_data.drop(['user_id'], axis=1)

used_feature = np.array(feature)
importance_feature = [0, 1, 2, 4, 5, 7, 8, 10, 13, 16, 17, 34, 35, 37, 40, 43, 47, 48, 52, 53, 54, 55, 76, 77, 78, 87,
                      93, 99, 105, 111, 113, 114, 124, 126, 128, 141, 151, 152, 156, 162, 163, 167, 171, 178]

used_feature = used_feature[np.array(importance_feature)]
# print(used_feature)
# X = X[used_feature]
# _valid_data = _valid_data[used_feature]
# test = test[used_feature]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

lgb_train = lgb.Dataset(X, y)
# lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 17,
    'max_depth': 8,
    'learning_rate': 0.5,
    # 'feature_fraction': 0.8,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,
    'verbose': 0,
    # 'subsample': 0.9,
    # 'colsample_bytree': 0.9,
    # 'num_boost_round': 100,
    'lambda_l1': 0.2,
    'lambda_l2': 75,
    'drop_rate': 0.2,
    'max_cat_threshold': 32,
    'min_data_in_leaf': 10,

    # 'num_trees':100
    # 'metric': {'auc', 'binary_logloss'}
    # 'task': 'train',
    # 'boosting_type': 'gbdt',
    # 'objective': 'binary',
    # 'metric': {'l2', 'auc'},
    # 'num_leaves': 17,
    # 'max_depth': 8,
    # 'learning_rate': 0.1,
    # # 'feature_fraction': 0.8,
    # 'bagging_fraction': 0.9,
    # 'bagging_freq': 500,
    # 'verbose': 0,
    # # 'subsample': 0.9,
    # # 'colsample_bytree': 0.9,
    # # 'num_boost_round': 100,
    # 'lambda_l1': 0.5,
    # 'lambda_l2': 80,
    # 'drop_rate': 0.5,
    # 'max_cat_threshold': 64,
    # 'min_data_in_leaf': 10,
    # 'num_trees':100
    # 'metric': {'auc', 'binary_logloss'}
}

grid_params = {
    'learning_rate': [0.01, 0.05, 0.1],  # 得到最佳参数0.01，
    'feature_fraction': [0.7, 0.8, 0.9],  # 如同学习率
    'bagging_fraction': [0.7, 0.8, 0.9],
    'num_leaves': [6, 7, 8, 9],
    'bagging_freq': [1, 3, 5, 10],
    'max_depth': [25, 50, 75],
}
#
# GridSearch code

# clf = lgb.LGBMClassifier(silent=False)
# grid = GridSearchCV(clf,grid_params,cv=3,scoring="roc_auc", verbose=5)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# print("Accuracy:{0:.1f}%".format(100*grid.best_score_))

print('开始训练......')
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=500,
                # categorical_feature=[1],
                )

pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
thre = 0.5
pred[pred > thre] = 1
pred[pred < thre] = 0
print(f'AUC：{metrics.roc_auc_score(y_test, pred)}')
print(f'Acc: {metrics.accuracy_score(y_test, pred)}')
print(f'Rec: {metrics.recall_score(y_test, pred)}')
print(f'F1：{metrics.f1_score(y_test, pred)}')
print(f'特征重要性：{list(gbm.feature_importance())}')
index = []
for i, f in enumerate(list(gbm.feature_importance())):
    if f >= 10:
        index.append(i)
print(index)

# valid
valid_label = gbm.predict(_valid_data)
valid_label2df = pd.DataFrame({'valid_label': valid_label})
valid = pd.concat([valid_data, valid_label2df], axis=1)

# lgb 预测的是概率
pre_ids = set(valid[valid['valid_label'] >= 0.50]['user_id'])
rel_ids = set(valid_data[valid_data['label'] == 1]['user_id'])

Precision = len(pre_ids.intersection(rel_ids)) / len(pre_ids)
Recall = len(pre_ids.intersection(rel_ids)) / len(rel_ids)
F1 = 2 * Precision * Recall / (Precision + Recall)
print('validation score...')
print(f"Precision: {Precision}\nRecall:{Recall}\nF1:{F1}")
#

# # submit
submit_pred = gbm.predict(test)
pred_label = pd.DataFrame({'label': submit_pred})
submit = pd.concat([test_data, pred_label], axis=1)
submit = submit[submit['label'] >= 0.50]['user_id']
print(submit.shape)
submit.to_csv(os.path.join(path, 'submission14.csv'), encoding='utf-8', index=None, header=None)
