import catboost as cab
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import os

path = os.path.join(os.path.curdir, 'data')

df = pd.read_csv(os.path.join(path, 'train_data/data223.csv'), encoding='utf-8')

X = df.drop(['user_id', 'label'], axis=1)
y = df['label']
# print(y)
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
eval_set = [(X_test, y_test)]

# # --------------------catboost--------------------- #
#
catparams = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.01,
    'loss_function': 'Logloss',
    'logging_level': 'Verbose',
    'thread_count': 3,
    'bagging_temperature':0.8,
    'l2_leaf_reg':4,
    'rsm':0.8,
    'random_seed':10086,
}
#
# # grid search
# # cat_grid_params={
# #     'iterations':[10,20,200],
# #     'depth':[3,6,7,9],
# #     'learning_rate':[0.005,0.01,0.1],
# # }
# # cat_clf=CatBoostClassifier(**catparams)
# # cat_grid=GridSearchCV(cat_clf,cat_grid_params)
# # cat_grid.fit(X_train,y_train)
# # print(cat_grid.best_params_)
# # print("Accuracy:{0:.1f}%".format(100*cat_grid.best_score_))
#

# model code
model2 = CatBoostClassifier(**catparams)
model2.fit(X_train, y_train, cat_features=[1])
y_pred2 = model2.predict(X_test)
print(y_pred2)
print(f'AUC：{metrics.roc_auc_score(y_test, y_pred2)}')
print(f'Acc: {metrics.accuracy_score(y_test, y_pred2)}')
print(f'Rec: {metrics.recall_score(y_test, y_pred2)}')
print(f'F1：{metrics.f1_score(y_test, y_pred2)}')

# valid
valid_data = pd.read_csv(os.path.join(path, 'train_data/data323.csv'), encoding='utf-8')
_valid_data = valid_data.drop(['user_id','label'], axis=1)
valid_label = model2.predict(_valid_data)
valid_label2df = pd.DataFrame({'valid_label': valid_label})
valid = pd.concat([valid_data, valid_label2df], axis=1)
pre_ids = set(valid[valid['valid_label'] == 1]['user_id'])
rel_ids = set(valid_data[valid_data['label'] == 1]['user_id'])

Precision = len(pre_ids.intersection(rel_ids)) / len(pre_ids)
Recall = len(pre_ids.intersection(rel_ids)) / len(rel_ids)
F1 = 2 * Precision * Recall / (Precision + Recall)

print(f"Precision: {Precision}\nRecall:{Recall}\nF1:{F1}")
