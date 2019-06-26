#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv('Data.csv')#读取csv文件
X = dataset.iloc[ : , :-1].values#.iloc[行，列]
Y = dataset.iloc[ : , 3].values  # : 全部行 or 列；[a]第a行 or 列
                                 # [a,b,c]第 a,b,c 行 or 列


# In[3]:


print(X);
print(Y);


# In[4]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)# 建立替换规则：将值为Nan的缺失值以均值做替换
imput = imputer.fit(X[ : , 1:3])# 应用模型规则
X[ : , 1:3] = imput.transform(X[ : , 1:3])


# # imputer
# 
# 可选参数
# 
#     strategy:  'mean'(默认的)， ‘median’中位数，‘most_frequent’出现频率最大的数
#     axis:  0(默认)， 1
#     copy: True(默认)，  False
# 输出
# 
#     numpy数组，之后可转化为DataFrame形式 
# 属性： 
# 
#     Imputer.statistics_可以查看每列的均值/中位数
# 特别说明：最好将imputer应用于整个数据集。因为虽然现在可能只有某一个属性存在缺失值，但是在新的数据中（如测试集）可能其他的属性也存在缺失值
# 
# [详解](http://www.dataivy.cn/blog/3-1-%e6%95%b0%e6%8d%ae%e6%b8%85%e6%b4%97%ef%bc%9a%e7%bc%ba%e5%a4%b1%e5%80%bc%e3%80%81%e5%bc%82%e5%b8%b8%e5%80%bc%e5%92%8c%e9%87%8d%e5%a4%8d%e5%80%bc%e7%9a%84%e5%a4%84%e7%90%86-2%e4%bb%a3%e7%a0%81/#more-1075)

# In[5]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()#即将离散型的数据转换成 00 到 n−1n−1 之间的数，这里 nn 是一个列表的不同取值的个数，可以认为是某个特征的所有不同取值的个数。
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])


# In[6]:


print(X)


# In[7]:


onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)


# [关于sklearn中的OneHotEncoder](https://www.jianshu.com/p/39855db1ed0b)
# 
# [OneHotEncoder独热编码和 LabelEncoder标签编码](https://www.cnblogs.com/king-lps/p/7846414.html)
# 
# [scikit-learn 中 OneHotEncoder 解析](https://www.cnblogs.com/zhoukui/p/9159909.html)

# In[8]:


print(X);


# In[9]:


print(Y);


# In[10]:


#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


# [train_test_split用法](https://blog.csdn.net/mrxjh/article/details/78481578)

# In[11]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)#归一化
X_test = sc_X.transform(X_test)


# In[12]:


print(X_train);
print(X_test);


# In[ ]:




