import xgboost as xgb
from xgboost import plot_importance
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
train = pd.read_csv(os.path.join(path, 'train_data/data123_323.csv'), encoding='utf-8')
X = train.drop(['user_id', 'label'], axis=1)
y = train['label']
# valid
valid_data = pd.read_csv(os.path.join(path, 'train_data/data223.csv'), encoding='utf-8')
_valid_data = valid_data.drop(['user_id', 'label'], axis=1)
# submit
test_data = pd.read_csv(os.path.join(path, 'train_data/data423.csv'), encoding='utf-8')
test = test_data.drop(['user_id'], axis=1)

used_feature = np.array(feature)
importance_feature = [0, 1, 2, 4, 5, 7, 8, 10, 13, 15, 16, 17, 35, 37, 38, 39, 40, 43, 47, 48, 51, 52, 53, 54, 55, 56,
                      61, 64, 66, 69, 71, 72, 74, 76, 78, 79, 93, 99, 105, 111, 112, 113, 114, 124, 126, 127, 128, 132,
                      136, 139, 141, 142, 143, 147, 151, 152, 153, 154, 155, 156, 157, 158, 160, 162, 163, 164, 165,
                      166, 167, 168, 169, 170, 171, 172, 173, 175, 177, 178, 179, 180, 181]

importance_feature1 = [0, 1, 2, 4, 5, 7, 8, 10, 16, 17, 35, 37, 38, 40, 47, 48, 51, 52, 53, 54, 55, 61, 71, 74, 76, 77,
                      78, 79, 93, 99, 105, 111, 112, 113, 114, 124, 126, 127, 128, 132, 136, 139, 141, 142, 143, 147,
                      151, 152, 153, 154, 155, 156, 157, 158, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                      172, 173, 175, 177, 178, 179, 180, 181]

# used_feature = used_feature[np.array(importance_feature)]
# print(used_feature)
# X = X[used_feature]
# _valid_data = _valid_data[used_feature]
# test = test[used_feature]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
eval_set = [(X_test, y_test)]

params = {
    # 'objective': 'multi:softmax',  # 多分类的问题
    # 'num_class': 10,               # 类别数，与 multisoftmax 并用
    'gamma': 9,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'iterations': 100,
    'lambda': 50,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 71.5,
    'learning_rate': 0.5,
    'n_estimators': 100,
    'num_round': 50,
    'reg_alpha':8,
    'reg_lambda':0.8,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,  # 在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    'booster': 'gbtree',
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.01,  # 如同学习率
    'seed': 1000,
    'nthread': 4,  # cpu 线程数
}
#
# grid_params = {
#     'learning_rate':[0.01,0.05,0.1],#得到最佳参数0.01，
#     'eta': [0.005,0.01 ] ,          # 如同学习率
#     'max_depth': [5,6,7,8,12],
#     'iterations': [1,3,5,10],
#     'min_child_weight': [0.1,0.5,1,2],
#     'n_estimators': [100,200,500,],
#     'num_round': [1,2,3,4],
# }
# #
# # GridSearch code
#
# clf = xgb.XGBClassifier(**params)
# grid = GridSearchCV(clf,grid_params)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# print("Accuracy:{0:.1f}%".format(100*grid.best_score_))


# model code
model1 = xgb.XGBClassifier(**params)
model1.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred1 = model1.predict(X_test)

acc1 = metrics.accuracy_score(y_test, y_pred1)
print(f'AUC: {metrics.roc_auc_score(y_test, y_pred1)}')
print(f'Acc: {acc1}')
print(f'Rec: {metrics.recall_score(y_test, y_pred1)}')
print(f'F1: {metrics.f1_score(y_test, y_pred1)}')
print(f'特征重要性：{list(model1.feature_importances_)}')
index = []
for i, f in enumerate(list(model1.feature_importances_)):
    if f >= 0.01:
        index.append(i)
print(index)

plot_importance(model1)
# plt.show()
print(model1.get_booster().get_fscore().keys())

# valid
valid_label = model1.predict(_valid_data)
valid_label2df = pd.DataFrame({'valid_label': valid_label})
valid = pd.concat([valid_data, valid_label2df], axis=1)
pre_ids = set(valid[valid['valid_label'] == 1]['user_id'])
rel_ids = set(valid_data[valid_data['label'] == 1]['user_id'])

Precision = len(pre_ids.intersection(rel_ids)) / len(pre_ids)
Recall = len(pre_ids.intersection(rel_ids)) / len(rel_ids)
F1 = 2 * Precision * Recall / (Precision + Recall)

print(f"Precision: {Precision}\nRecall:{Recall}\nF1:{F1}")
#
# submit
submit_pred = model1.predict(test)
pred_label = pd.DataFrame({'label': submit_pred})
submit = pd.concat([test_data, pred_label], axis=1)
submit = submit[submit['label'] == 1]['user_id']
print(submit.shape)
submit.to_csv(os.path.join(path,'submission12.csv'),encoding='utf-8',index=None,header=None)
