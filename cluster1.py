
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r'E:\competition\kuaishou\data\data.csv',index_col='user_id').\
    drop(['day_range','day_range_y','day_range_y','label','label_x','label_y'],axis=1)
# X=pd.read_csv(r'E:\competition\kuaishou\data\data.csv')
X=Imputer().fit_transform(df)
X=pd.DataFrame(X)
print(X.shape)
kmodel=KMeans(n_clusters=2,init='random',max_iter=1000)
kmodel.fit(X)

result = kmodel.fit_predict(X)

print ("Predicting result: ", result)
r1 = pd.Series(kmodel.labels_).value_counts()  #统计各个类别的数目
print(r1)
r2 = pd.DataFrame(kmodel.cluster_centers_)     #找出聚类中心
# print(r2)
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(X.columns) + [u'类别数目'] #重命名表头
print(r)











